using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace MuseTalk.Core
{
    using Utils;
    using Models;

    /// <summary>
    /// LivePortrait prediction state for frame-by-frame inference
    /// </summary>
    public class LivePortraitPredInfo
    {
        public Vector2[] Landmarks { get; set; }
        public MotionInfo InitialMotionInfo { get; set; }
    }

    /// <summary>
    /// LivePortrait inference result
    /// </summary>
    public class LivePortraitResult
    {
        public bool Success { get; set; }
        public List<Texture2D> GeneratedFrames { get; set; }
        public string ErrorMessage { get; set; }
        
        public LivePortraitResult()
        {
            GeneratedFrames = new List<Texture2D>();
        }
    }

    /// <summary>
    /// LivePortrait input configuration
    /// </summary>
    public class LivePortraitInput
    {
        public Texture2D SourceImage { get; set; }
        public Texture2D[] DrivingFrames { get; set; }
        public bool UseComposite { get; set; } = false;
    }

    /// <summary>
    /// Core LivePortrait inference engine that matches onnx_inference.py exactly
    /// Generates talking head animations from a single source image and driving video frames
    /// </summary>
    public class LivePortraitInference : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        // LivePortrait ONNX models
        private AppearanceFeatureExtractorModel _appearanceExtractor;
        private MotionExtractorModel _motionExtractor;
        private WarpingSPADEModel _warpingSpade;
        private StitchingModel _stitching;
        private LivePortraitLandmarkModel _landmarkRunner;
        private Landmark106Model _landmark106;
        
        // Face analysis (reuse existing InsightFace from MuseTalk)
        private InsightFaceHelper _insightFaceHelper;
        
        // Configuration and state
        private MuseTalkConfig _config;
        private bool _initialized = false;
        private bool _disposed = false;
        
        // Mask template for blending
        private Texture2D _maskTemplate;
        
        public bool IsInitialized => _initialized;
        
        /// <summary>
        /// Initialize LivePortrait inference with MuseTalk configuration
        /// </summary>
        public LivePortraitInference(MuseTalkConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            
            try
            {
                InitializeModels();
                LoadMaskTemplate();
                _initialized = true;
                Logger.Log("[LivePortraitInference] Successfully initialized");
            }
            catch (Exception e)
            {
                Logger.LogError($"[LivePortraitInference] Failed to initialize: {e.Message}");
                _initialized = false;
            }
        }
        
        /// <summary>
        /// Initialize all LivePortrait models
        /// </summary>
        private void InitializeModels()
        {
            Logger.Log("[LivePortraitInference] Initializing LivePortrait models...");
            
            _appearanceExtractor = new AppearanceFeatureExtractorModel(_config);
            _motionExtractor = new MotionExtractorModel(_config);
            _warpingSpade = new WarpingSPADEModel(_config);
            _stitching = new StitchingModel(_config);
            _landmarkRunner = new LivePortraitLandmarkModel(_config);
            _landmark106 = new Landmark106Model(_config);
            
            // Reuse existing InsightFace helper from MuseTalk
            _insightFaceHelper = new InsightFaceHelper(_config);
            
            // Verify all models initialized
            bool allInitialized = _appearanceExtractor.IsInitialized &&
                                 _motionExtractor.IsInitialized &&
                                 _warpingSpade.IsInitialized &&
                                 _stitching.IsInitialized &&
                                 _landmarkRunner.IsInitialized &&
                                 _landmark106.IsInitialized &&
                                 _insightFaceHelper.IsInitialized;
            
            if (!allInitialized)
            {
                var failedModels = new List<string>();
                if (!_appearanceExtractor.IsInitialized) failedModels.Add("AppearanceExtractor");
                if (!_motionExtractor.IsInitialized) failedModels.Add("MotionExtractor");
                if (!_warpingSpade.IsInitialized) failedModels.Add("WarpingSPADE");
                if (!_stitching.IsInitialized) failedModels.Add("Stitching");
                if (!_landmarkRunner.IsInitialized) failedModels.Add("LandmarkRunner");
                if (!_landmark106.IsInitialized) failedModels.Add("Landmark106");
                if (!_insightFaceHelper.IsInitialized) failedModels.Add("InsightFace");
                
                throw new InvalidOperationException($"Failed to initialize models: {string.Join(", ", failedModels)}");
            }
            
            Logger.Log("[LivePortraitInference] All models initialized successfully");
        }
        
        /// <summary>
        /// Load mask template for blending
        /// </summary>
        private void LoadMaskTemplate()
        {
            try
            {
                var maskPath = System.IO.Path.Combine(_config.ModelPath, "mask_template.png");
                if (System.IO.File.Exists(maskPath))
                {
                    var maskBytes = System.IO.File.ReadAllBytes(maskPath);
                    _maskTemplate = new Texture2D(2, 2);
                    _maskTemplate.LoadImage(maskBytes);
                    Logger.Log("[LivePortraitInference] Mask template loaded successfully");
                }
                else
                {
                    Logger.LogWarning($"[LivePortraitInference] Mask template not found at: {maskPath}");
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[LivePortraitInference] Failed to load mask template: {e.Message}");
            }
        }
        
        /// <summary>
        /// Generate talking head animation from source image and driving frames
        /// Matches Python LivePortraitWrapper.execute exactly
        /// </summary>
        public async Task<LivePortraitResult> GenerateAsync(LivePortraitInput input)
        {
            if (!_initialized)
                throw new InvalidOperationException("LivePortrait inference not initialized");
                
            if (input?.SourceImage == null || input.DrivingFrames == null || input.DrivingFrames.Length == 0)
                throw new ArgumentException("Invalid input: source image and driving frames are required");
                
            try
            {
                Logger.Log($"[LivePortraitInference] === STARTING LIVEPORTRAIT GENERATION ===");
                Logger.Log($"[LivePortraitInference] Source: {input.SourceImage.width}x{input.SourceImage.height}, Driving frames: {input.DrivingFrames.Length}");
                
                // Step 1: Process source image - matches Python crop_src_image
                Logger.Log("[LivePortraitInference] STAGE 1: Processing source image...");
                var sourceData = await ProcessSourceImage(input.SourceImage);
                Logger.Log("[LivePortraitInference] Stage 1 completed - Source processed");
                
                // Step 2: Generate frames - matches Python frame-by-frame prediction
                Logger.Log("[LivePortraitInference] STAGE 2: Generating frames...");
                var frames = await GenerateFrames(sourceData, input.DrivingFrames, input.UseComposite);
                Logger.Log($"[LivePortraitInference] Stage 2 completed - Generated {frames.Count} frames");
                
                var result = new LivePortraitResult
                {
                    Success = true,
                    GeneratedFrames = frames
                };
                
                Logger.Log($"[LivePortraitInference] === GENERATION COMPLETED ===");
                return result;
            }
            catch (Exception e)
            {
                Logger.LogError($"[LivePortraitInference] Generation failed: {e.Message}");
                return new LivePortraitResult
                {
                    Success = false,
                    ErrorMessage = e.Message
                };
            }
        }
        
        /// <summary>
        /// Process source image, matching Python crop_src_image exactly
        /// </summary>
        private async Task<SourceData> ProcessSourceImage(Texture2D sourceImage)
        {
            // Step 1: Preprocess source image (matching Python src_preprocess)
            var preprocessedSource = PreprocessSourceImage(sourceImage);
            
            // Step 2: Detect face and get initial landmarks
            var faceAnalysisResult = AnalyzeFace(preprocessedSource);
            if (!faceAnalysisResult.HasFace)
            {
                throw new InvalidOperationException("No face detected in source image");
            }
            
            // Step 3: Get 106 landmarks
            var landmarks106 = _landmark106.GetLandmarks(preprocessedSource, faceAnalysisResult.BoundingBox);
            
            // Step 4: Crop source image (matching Python crop_image)
            var cropInfo = CropSourceImage(preprocessedSource, landmarks106);
            
            // Step 5: Get enhanced landmarks using landmark runner
            var enhancedLandmarks = await Task.Run(() => 
                _landmarkRunner.GetLandmarks(preprocessedSource, landmarks106)
            );
            
            // Step 6: Prepare source data (matching Python prepare_source)
            var sourceData = await PrepareSourceData(cropInfo, enhancedLandmarks);
            
            return sourceData;
        }
        
        /// <summary>
        /// Preprocess source image, matching Python src_preprocess exactly
        /// </summary>
        private Texture2D PreprocessSourceImage(Texture2D source)
        {
            int h = source.height;
            int w = source.width;
            
            // Adjust size according to maximum dimension
            const int maxDim = 1280;
            if (Mathf.Max(h, w) > maxDim)
            {
                int newH, newW;
                if (h > w)
                {
                    newH = maxDim;
                    newW = Mathf.RoundToInt(w * (maxDim / (float)h));
                }
                else
                {
                    newW = maxDim;
                    newH = Mathf.RoundToInt(h * (maxDim / (float)w));
                }
                source = TextureUtils.ResizeTexture(source, newW, newH);
                h = newH;
                w = newW;
            }
            
            // Ensure dimensions are multiples of 2
            const int division = 2;
            int newHeight = h - (h % division);
            int newWidth = w - (w % division);
            
            if (newHeight == 0 || newWidth == 0)
                return source; // No need to process
                
            if (newHeight != h || newWidth != w)
            {
                var croppedPixels = source.GetPixels(0, 0, newWidth, newHeight);
                var croppedTexture = new Texture2D(newWidth, newHeight, TextureFormat.RGB24, false);
                croppedTexture.SetPixels(croppedPixels);
                croppedTexture.Apply();
                
                if (source != null) UnityEngine.Object.DestroyImmediate(source);
                return croppedTexture;
            }
            
            return source;
        }
        
        /// <summary>
        /// Analyze face using existing InsightFace, matching Python face_analysis
        /// </summary>
        private FaceAnalysisResult AnalyzeFace(Texture2D image)
        {
            var result = _insightFaceHelper.GetLandmarkAndBbox(new[] { image });
            var coords = result.Item1;
            
            if (coords.Count == 0 || coords[0] == InsightFaceHelper.CoordPlaceholder)
            {
                return new FaceAnalysisResult { HasFace = false };
            }
            
            var bbox = coords[0];
            return new FaceAnalysisResult
            {
                HasFace = true,
                BoundingBox = new Rect(bbox.x, bbox.y, bbox.z - bbox.x, bbox.w - bbox.y)
            };
        }
        
        /// <summary>
        /// Crop source image based on landmarks, matching Python crop_image
        /// </summary>
        private CropInfo CropSourceImage(Texture2D image, Vector2[] landmarks)
        {
            // Calculate crop parameters using landmarks
            var (cropRect, transform) = CalculateCropParameters(landmarks, image.width, image.height);
            
            // Perform crop
            var croppedImage = TextureUtils.CropTexture(image, cropRect);
            var resizedCrop = TextureUtils.ResizeTexture(croppedImage, 512, 512);
            var resizedCrop256 = TextureUtils.ResizeTexture(croppedImage, 256, 256);
            
            UnityEngine.Object.DestroyImmediate(croppedImage);
            
            return new CropInfo
            {
                ImageCrop = resizedCrop,
                ImageCrop256x256 = resizedCrop256,
                Transform = transform,
                CropRect = cropRect
            };
        }
        
        /// <summary>
        /// Calculate crop parameters, matching Python crop_image logic
        /// </summary>
        private (Rect, Matrix4x4) CalculateCropParameters(Vector2[] landmarks, int imageWidth, int imageHeight)
        {
            // Find landmark bounds
            float minX = landmarks.Min(p => p.x);
            float maxX = landmarks.Max(p => p.x);
            float minY = landmarks.Min(p => p.y);
            float maxY = landmarks.Max(p => p.y);
            
            // Calculate center and size
            Vector2 center = new Vector2((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);
            float width = maxX - minX;
            float height = maxY - minY;
            
            // Apply scale and offset (matching Python parameters)
            const float scale = 2.3f;
            const float vyRatio = -0.125f;
            
            float cropSize = Mathf.Max(width, height) * scale;
            center.y += height * vyRatio;
            
            // Create crop rectangle
            float x = Mathf.Max(0, center.x - cropSize * 0.5f);
            float y = Mathf.Max(0, center.y - cropSize * 0.5f);
            float w = Mathf.Min(imageWidth - x, cropSize);
            float h = Mathf.Min(imageHeight - y, cropSize);
            
            var cropRect = new Rect(x, y, w, h);
            
            // Create transform matrix
            var transform = Matrix4x4.identity;
            transform.m00 = w / 512f;
            transform.m11 = h / 512f;
            transform.m03 = x;
            transform.m13 = y;
            
            return (cropRect, transform);
        }
        
        /// <summary>
        /// Prepare source data for inference, matching Python prepare_source
        /// </summary>
        private async Task<SourceData> PrepareSourceData(CropInfo cropInfo, Vector2[] landmarks)
        {
            return await Task.Run(() =>
            {
                // Extract appearance features
                var appearanceFeatures = _appearanceExtractor.ExtractFeature3D(cropInfo.ImageCrop256x256);
                
                // Extract motion info
                var motionInfo = _motionExtractor.GetMotionInfo(cropInfo.ImageCrop256x256);
                
                // Calculate rotation matrix
                var rotationMatrix = CalculateRotationMatrix(motionInfo.Pitch[0], motionInfo.Yaw[0], motionInfo.Roll[0]);
                
                // Transform keypoints
                var transformedKeypoints = TransformKeypoints(motionInfo);
                
                return new SourceData
                {
                    CropInfo = cropInfo,
                    Landmarks = landmarks,
                    AppearanceFeatures = appearanceFeatures,
                    MotionInfo = motionInfo,
                    RotationMatrix = rotationMatrix,
                    TransformedKeypoints = transformedKeypoints
                };
            });
        }
        
        /// <summary>
        /// Calculate rotation matrix from Euler angles, matching Python get_rotation_matrix
        /// </summary>
        private float[,] CalculateRotationMatrix(float pitch, float yaw, float roll)
        {
            // Convert degrees to radians
            float p = pitch * Mathf.Deg2Rad;
            float y = yaw * Mathf.Deg2Rad;
            float r = roll * Mathf.Deg2Rad;
            
            // Calculate rotation matrix components
            float cosP = Mathf.Cos(p), sinP = Mathf.Sin(p);
            float cosY = Mathf.Cos(y), sinY = Mathf.Sin(y);
            float cosR = Mathf.Cos(r), sinR = Mathf.Sin(r);
            
            // Combined rotation matrix (ZYX order)
            var matrix = new float[3, 3];
            matrix[0, 0] = cosY * cosR;
            matrix[0, 1] = cosY * sinR;
            matrix[0, 2] = -sinY;
            matrix[1, 0] = sinP * sinY * cosR - cosP * sinR;
            matrix[1, 1] = sinP * sinY * sinR + cosP * cosR;
            matrix[1, 2] = sinP * cosY;
            matrix[2, 0] = cosP * sinY * cosR + sinP * sinR;
            matrix[2, 1] = cosP * sinY * sinR - sinP * cosR;
            matrix[2, 2] = cosP * cosY;
            
            return matrix;
        }
        
        /// <summary>
        /// Transform keypoints, matching Python transform_keypoint
        /// </summary>
        private float[] TransformKeypoints(MotionInfo motionInfo)
        {
            // Reshape keypoints to proper format [batch, num_kp, 3]
            int numKeypoints = motionInfo.Keypoints.Length / 3;
            var transformedKp = new float[numKeypoints * 3];
            
            for (int i = 0; i < numKeypoints; i++)
            {
                transformedKp[i * 3] = motionInfo.Keypoints[i * 3];
                transformedKp[i * 3 + 1] = motionInfo.Keypoints[i * 3 + 1];
                transformedKp[i * 3 + 2] = motionInfo.Keypoints[i * 3 + 2];
            }
            
            return transformedKp;
        }
        
        /// <summary>
        /// Generate frames from source and driving data, matching Python predict
        /// </summary>
        private async Task<List<Texture2D>> GenerateFrames(SourceData sourceData, Texture2D[] drivingFrames, bool useComposite)
        {
            var frames = new List<Texture2D>();
            var predInfo = new LivePortraitPredInfo();
            
            for (int frameId = 0; frameId < drivingFrames.Length; frameId++)
            {
                try
                {
                    var frame = await PredictFrame(frameId, sourceData, drivingFrames[frameId], predInfo, useComposite);
                    frames.Add(frame);
                    
                    if (frameId % 10 == 0)
                    {
                        Logger.Log($"[LivePortraitInference] Processed frame {frameId + 1}/{drivingFrames.Length}");
                    }
                }
                catch (Exception e)
                {
                    Logger.LogError($"[LivePortraitInference] Failed to process frame {frameId}: {e.Message}");
                    // Continue with next frame
                }
            }
            
            return frames;
        }
        
        /// <summary>
        /// Predict single frame, matching Python predict function exactly
        /// </summary>
        private async Task<Texture2D> PredictFrame(int frameId, SourceData sourceData, Texture2D drivingFrame, LivePortraitPredInfo predInfo, bool useComposite)
        {
            return await Task.Run(() =>
            {
                // Step 1: Calculate landmarks from driving frame (matching calc_lmks_from_cropped_video)
                bool isFirstFrame = predInfo.Landmarks == null;
                Vector2[] currentLandmarks;
                
                if (isFirstFrame)
                {
                    var faceResult = AnalyzeFace(drivingFrame);
                    if (!faceResult.HasFace)
                    {
                        throw new InvalidOperationException($"No face detected in driving frame {frameId}");
                    }
                    currentLandmarks = _landmark106.GetLandmarks(drivingFrame, faceResult.BoundingBox);
                    currentLandmarks = _landmarkRunner.GetLandmarks(drivingFrame, currentLandmarks);
                }
                else
                {
                    currentLandmarks = _landmarkRunner.GetLandmarks(drivingFrame, predInfo.Landmarks);
                }
                predInfo.Landmarks = currentLandmarks;
                
                // Step 2: Calculate driving ratios (matching calc_driving_ratio)
                var (eyesRatio, lipRatio) = CalculateDrivingRatios(currentLandmarks);
                
                // Step 3: Prepare driving frame (matching prepare_driving_videos)
                var resizedDriving = TextureUtils.ResizeTexture(drivingFrame, 256, 256);
                var drivingMotionInfo = _motionExtractor.GetMotionInfo(resizedDriving);
                var drivingRotationMatrix = CalculateRotationMatrix(drivingMotionInfo.Pitch[0], drivingMotionInfo.Yaw[0], drivingMotionInfo.Roll[0]);
                
                if (isFirstFrame)
                {
                    predInfo.InitialMotionInfo = drivingMotionInfo;
                }
                
                // Step 4: Calculate new motion parameters (matching Python logic)
                var newMotionParams = CalculateNewMotionParameters(sourceData, drivingMotionInfo, predInfo.InitialMotionInfo);
                
                // Step 5: Apply stitching
                var stitchedKeypoints = _stitching.Stitch(sourceData.TransformedKeypoints, newMotionParams.NewKeypoints);
                
                // Step 6: Warp image using SPADE
                var warpedOutput = _warpingSpade.WarpImage(sourceData.AppearanceFeatures, sourceData.TransformedKeypoints, stitchedKeypoints);
                
                // Step 7: Convert to texture and apply blending
                var outputTexture = ConvertArrayToTexture(warpedOutput);
                
                // Step 8: Apply paste back or composite (matching Python logic)
                Texture2D finalFrame;
                if (useComposite)
                {
                    finalFrame = CreateCompositeFrame(drivingFrame, sourceData.CropInfo.ImageCrop256x256, outputTexture);
                }
                else
                {
                    finalFrame = PasteBack(outputTexture, sourceData, drivingFrame);
                }
                
                UnityEngine.Object.DestroyImmediate(resizedDriving);
                UnityEngine.Object.DestroyImmediate(outputTexture);
                
                return finalFrame;
            });
        }
        
        /// <summary>
        /// Calculate driving ratios for eyes and lips, matching Python calc_driving_ratio
        /// </summary>
        private (float[], float[]) CalculateDrivingRatios(Vector2[] landmarks)
        {
            // Calculate eye ratios (matching Python indices)
            var eyesRatio = CalculateDistanceRatio(landmarks, 6, 18, 0, 12);
            var eyesRatio2 = CalculateDistanceRatio(landmarks, 30, 42, 24, 36);
            var combinedEyesRatio = new float[eyesRatio.Length + eyesRatio2.Length];
            Array.Copy(eyesRatio, 0, combinedEyesRatio, 0, eyesRatio.Length);
            Array.Copy(eyesRatio2, 0, combinedEyesRatio, eyesRatio.Length, eyesRatio2.Length);
            
            // Calculate lip ratio
            var lipRatio = CalculateDistanceRatio(landmarks, 90, 102, 48, 66);
            
            return (combinedEyesRatio, lipRatio);
        }
        
        /// <summary>
        /// Calculate distance ratio between landmark points, matching Python calculate_distance_ratio
        /// </summary>
        private float[] CalculateDistanceRatio(Vector2[] landmarks, int idx1, int idx2, int idx3, int idx4)
        {
            if (landmarks.Length <= Mathf.Max(idx1, idx2, idx3, idx4))
                return new float[] { 1.0f };
                
            float dist1 = Vector2.Distance(landmarks[idx1], landmarks[idx2]);
            float dist2 = Vector2.Distance(landmarks[idx3], landmarks[idx4]);
            
            float ratio = dist2 > 0 ? dist1 / dist2 : 1.0f;
            return new float[] { ratio };
        }
        
        /// <summary>
        /// Calculate new motion parameters, matching Python inference logic
        /// </summary>
        private NewMotionParameters CalculateNewMotionParameters(SourceData sourceData, MotionInfo drivingMotion, MotionInfo initialDrivingMotion)
        {
            // Calculate relative motion changes
            var deltaRotation = CalculateRelativeRotation(drivingMotion, initialDrivingMotion, sourceData.RotationMatrix);
            var deltaExpression = CalculateDeltaExpression(sourceData.MotionInfo, drivingMotion, initialDrivingMotion);
            var deltaScale = CalculateDeltaScale(sourceData.MotionInfo, drivingMotion, initialDrivingMotion);
            var deltaTranslation = CalculateDeltaTranslation(sourceData.MotionInfo, drivingMotion, initialDrivingMotion);
            
            // Apply transformations to source keypoints
            var newKeypoints = ApplyMotionTransformations(sourceData.TransformedKeypoints, deltaRotation, deltaExpression, deltaScale, deltaTranslation);
            
            return new NewMotionParameters
            {
                NewKeypoints = newKeypoints,
                DeltaRotation = deltaRotation,
                DeltaExpression = deltaExpression,
                DeltaScale = deltaScale,
                DeltaTranslation = deltaTranslation
            };
        }
        
        private float[,] CalculateRelativeRotation(MotionInfo driving, MotionInfo initialDriving, float[,] sourceRotation)
        {
            var drivingRot = CalculateRotationMatrix(driving.Pitch[0], driving.Yaw[0], driving.Roll[0]);
            var initialRot = CalculateRotationMatrix(initialDriving.Pitch[0], initialDriving.Yaw[0], initialDriving.Roll[0]);
            
            // Calculate R_new = (R_d @ R_d_0.T) @ R_s
            return MultiplyRotationMatrices(MultiplyRotationMatrices(drivingRot, TransposeMatrix(initialRot)), sourceRotation);
        }
        
        private float[] CalculateDeltaExpression(MotionInfo source, MotionInfo driving, MotionInfo initialDriving)
        {
            var delta = new float[source.Expression.Length];
            for (int i = 0; i < delta.Length; i++)
            {
                delta[i] = source.Expression[i] + (driving.Expression[i] - initialDriving.Expression[i]);
            }
            return delta;
        }
        
        private float[] CalculateDeltaScale(MotionInfo source, MotionInfo driving, MotionInfo initialDriving)
        {
            var delta = new float[source.Scale.Length];
            for (int i = 0; i < delta.Length; i++)
            {
                float drivingScale = initialDriving.Scale[i] != 0 ? driving.Scale[i] / initialDriving.Scale[i] : 1.0f;
                delta[i] = source.Scale[i] * drivingScale;
            }
            return delta;
        }
        
        private float[] CalculateDeltaTranslation(MotionInfo source, MotionInfo driving, MotionInfo initialDriving)
        {
            var delta = new float[source.Translation.Length];
            for (int i = 0; i < delta.Length; i++)
            {
                delta[i] = source.Translation[i] + (driving.Translation[i] - initialDriving.Translation[i]);
            }
            // Zero out Z translation (matching Python t_new[..., 2] = 0)
            if (delta.Length > 2)
                delta[2] = 0;
            return delta;
        }
        
        private float[] ApplyMotionTransformations(float[] sourceKeypoints, float[,] rotation, float[] deltaExp, float[] deltaScale, float[] deltaTranslation)
        {
            int numKeypoints = sourceKeypoints.Length / 3;
            var result = new float[sourceKeypoints.Length];
            
            for (int i = 0; i < numKeypoints; i++)
            {
                // Get source keypoint
                float x = sourceKeypoints[i * 3];
                float y = sourceKeypoints[i * 3 + 1];
                float z = sourceKeypoints[i * 3 + 2];
                
                // Apply rotation
                float newX = rotation[0, 0] * x + rotation[0, 1] * y + rotation[0, 2] * z;
                float newY = rotation[1, 0] * x + rotation[1, 1] * y + rotation[1, 2] * z;
                float newZ = rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2] * z;
                
                // Apply expression delta
                if (i * 3 < deltaExp.Length)
                {
                    newX += deltaExp[i * 3];
                    newY += deltaExp[i * 3 + 1];
                    newZ += deltaExp[i * 3 + 2];
                }
                
                // Apply scale
                if (deltaScale.Length > 0)
                {
                    float scale = deltaScale[0];
                    newX *= scale;
                    newY *= scale;
                    newZ *= scale;
                }
                
                // Apply translation
                if (deltaTranslation.Length >= 3)
                {
                    newX += deltaTranslation[0];
                    newY += deltaTranslation[1];
                    newZ += deltaTranslation[2];
                }
                
                result[i * 3] = newX;
                result[i * 3 + 1] = newY;
                result[i * 3 + 2] = newZ;
            }
            
            return result;
        }
        
        // Helper methods for matrix operations
        private float[,] MultiplyRotationMatrices(float[,] a, float[,] b)
        {
            var result = new float[3, 3];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    result[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j];
                }
            }
            return result;
        }
        
        private float[,] TransposeMatrix(float[,] matrix)
        {
            var result = new float[3, 3];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    result[i, j] = matrix[j, i];
                }
            }
            return result;
        }
        
        /// <summary>
        /// Convert float array to texture, matching Python tensor to image conversion
        /// </summary>
        private Texture2D ConvertArrayToTexture(float[] data)
        {
            // Assume output is [1, 3, H, W] format
            int channels = 3;
            int height = 256; // Typical LivePortrait output
            int width = 256;
            
            var texture = new Texture2D(width, height, TextureFormat.RGB24, false);
            var pixels = new Color[width * height];
            
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int pixelIndex = h * width + w;
                    int rIndex = 0 * height * width + h * width + w;
                    int gIndex = 1 * height * width + h * width + w;
                    int bIndex = 2 * height * width + h * width + w;
                    
                    float r = Mathf.Clamp01(data[rIndex]);
                    float g = Mathf.Clamp01(data[gIndex]);
                    float b = Mathf.Clamp01(data[bIndex]);
                    
                    pixels[pixelIndex] = new Color(r, g, b, 1.0f);
                }
            }
            
            texture.SetPixels(pixels);
            texture.Apply();
            return texture;
        }
        
        /// <summary>
        /// Create composite frame, matching Python concat_frame
        /// </summary>
        private Texture2D CreateCompositeFrame(Texture2D driving, Texture2D crop, Texture2D predicted)
        {
            int width = driving.width + crop.width + predicted.width;
            int height = Mathf.Max(driving.height, crop.height, predicted.height);
            
            var composite = new Texture2D(width, height, TextureFormat.RGB24, false);
            
            // Copy driving frame
            var drivingPixels = driving.GetPixels();
            composite.SetPixels(0, 0, driving.width, driving.height, drivingPixels);
            
            // Copy crop
            var cropPixels = crop.GetPixels();
            composite.SetPixels(driving.width, 0, crop.width, crop.height, cropPixels);
            
            // Copy predicted
            var predictedPixels = predicted.GetPixels();
            composite.SetPixels(driving.width + crop.width, 0, predicted.width, predicted.height, predictedPixels);
            
            composite.Apply();
            return composite;
        }
        
        /// <summary>
        /// Paste back predicted face to original image, matching Python paste_back
        /// </summary>
        private Texture2D PasteBack(Texture2D predicted, SourceData sourceData, Texture2D originalFrame)
        {
            // For now, return the predicted face resized to match original frame
            // This would need more sophisticated blending logic for production use
            return TextureUtils.ResizeTexture(predicted, originalFrame.width, originalFrame.height);
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _appearanceExtractor?.Dispose();
                _motionExtractor?.Dispose();
                _warpingSpade?.Dispose();
                _stitching?.Dispose();
                _landmarkRunner?.Dispose();
                _landmark106?.Dispose();
                _insightFaceHelper?.Dispose();
                
                if (_maskTemplate != null)
                    UnityEngine.Object.DestroyImmediate(_maskTemplate);
                
                _disposed = true;
                Logger.Log("[LivePortraitInference] Disposed");
            }
        }
    }
    
    // Supporting data structures
    public class FaceAnalysisResult
    {
        public bool HasFace { get; set; }
        public Rect BoundingBox { get; set; }
    }
    
    public class CropInfo
    {
        public Texture2D ImageCrop { get; set; }
        public Texture2D ImageCrop256x256 { get; set; }
        public Matrix4x4 Transform { get; set; }
        public Rect CropRect { get; set; }
    }
    
    public class SourceData
    {
        public CropInfo CropInfo { get; set; }
        public Vector2[] Landmarks { get; set; }
        public float[] AppearanceFeatures { get; set; }
        public MotionInfo MotionInfo { get; set; }
        public float[,] RotationMatrix { get; set; }
        public float[] TransformedKeypoints { get; set; }
    }
    
    public class NewMotionParameters
    {
        public float[] NewKeypoints { get; set; }
        public float[,] DeltaRotation { get; set; }
        public float[] DeltaExpression { get; set; }
        public float[] DeltaScale { get; set; }
        public float[] DeltaTranslation { get; set; }
    }
}
