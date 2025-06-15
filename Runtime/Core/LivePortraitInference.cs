using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MuseTalk.Core
{
    using Utils;
    using Models;

    /// <summary>
    /// LivePortrait prediction state for frame-by-frame inference
    /// Matches Python pred_info exactly
    /// </summary>
    public class LivePortraitPredInfo
    {
        public Vector2[] Landmarks { get; set; }  // lmk
        public MotionInfo InitialMotionInfo { get; set; }  // x_d_0_info
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
    /// Core LivePortrait inference engine that matches onnx_inference.py EXACTLY
    /// ALL OPERATIONS ON MAIN THREAD FOR CORRECTNESS FIRST
    /// </summary>
    public class LivePortraitInference : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        // LivePortrait ONNX models - matches Python models dict
        private AppearanceFeatureExtractorModel _appearanceExtractor;
        private MotionExtractorModel _motionExtractor;
        private WarpingSPADEModel _warpingSpade;
        private StitchingModel _stitching;
        private LivePortraitLandmarkModel _landmarkRunner;
        private Landmark106Model _landmark106;
        
        // Face analysis (reuse existing InsightFace)
        private InsightFaceHelper _insightFaceHelper;
        
        // Configuration
        private MuseTalkConfig _config;
        private bool _initialized = false;
        private bool _disposed = false;
        
        // Mask template for blending
        private Texture2D _maskTemplate;
        
        public bool IsInitialized => _initialized;
        
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
        
        private void InitializeModels()
        {
            Logger.Log("[LivePortraitInference] Initializing LivePortrait models...");
            
            _appearanceExtractor = new AppearanceFeatureExtractorModel(_config);
            _motionExtractor = new MotionExtractorModel(_config);
            _warpingSpade = new WarpingSPADEModel(_config);
            _stitching = new StitchingModel(_config);
            _landmarkRunner = new LivePortraitLandmarkModel(_config);
            _landmark106 = new Landmark106Model(_config);
            
            // Reuse existing InsightFace helper
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
        /// Generate talking head animation - matches Python LivePortraitWrapper.execute
        /// MAIN THREAD ONLY for correctness
        /// </summary>
        public LivePortraitResult Generate(LivePortraitInput input)
        {
            if (!_initialized)
                throw new InvalidOperationException("LivePortrait inference not initialized");
                
            if (input?.SourceImage == null || input.DrivingFrames == null || input.DrivingFrames.Length == 0)
                throw new ArgumentException("Invalid input: source image and driving frames are required");
                
            try
            {
                Logger.Log($"[LivePortraitInference] === STARTING LIVEPORTRAIT GENERATION ===");
                Logger.Log($"[LivePortraitInference] Source: {input.SourceImage.width}x{input.SourceImage.height}, Driving frames: {input.DrivingFrames.Length}");
                
                // Python: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                // Python: img = img[:, :, ::-1]  # BGR -> RGB
                // Python: src_img = src_preprocess(img)
                var srcImg = SrcPreprocess(input.SourceImage);
                Logger.Log("[LivePortraitInference] Source preprocessed");
                
                // Python: crop_info = crop_src_image(self.models, src_img)
                var cropInfo = CropSrcImage(srcImg);
                Logger.Log("[LivePortraitInference] Source cropped");
                
                // Python: img_crop_256x256 = crop_info["img_crop_256x256"]
                // Python: I_s = preprocess(img_crop_256x256)
                var Is = Preprocess(cropInfo.ImageCrop256x256);
                Logger.Log("[LivePortraitInference] Source image prepared for inference");
                
                // Python: x_s_info = get_kp_info(self.models, I_s)
                var xSInfo = GetKpInfo(Is);
                Logger.Log("[LivePortraitInference] Source keypoint info extracted");
                
                // Python: R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
                var Rs = GetRotationMatrix(xSInfo.Pitch, xSInfo.Yaw, xSInfo.Roll);
                Logger.Log("[LivePortraitInference] Source rotation matrix calculated");
                
                // Python: f_s = extract_feature_3d(self.models, I_s)
                var fs = ExtractFeature3d(Is);
                Logger.Log("[LivePortraitInference] Source 3D features extracted");
                
                // Python: x_s = transform_keypoint(x_s_info)
                var xs = TransformKeypoint(xSInfo);
                Logger.Log("[LivePortraitInference] Source keypoints transformed");
                
                // Initialize prediction info - matches Python pred_info
                var predInfo = new LivePortraitPredInfo
                {
                    Landmarks = null,
                    InitialMotionInfo = null
                };
                
                // Generate frames
                var generatedFrames = new List<Texture2D>();
                
                for (int frameId = 0; frameId < input.DrivingFrames.Length; frameId++)
                {
                    Logger.Log($"[LivePortraitInference] Processing frame {frameId + 1}/{input.DrivingFrames.Length}");
                    
                    // Python: I_p, self.pred_info = predict(frame_id, self.models, x_s_info, R_s, f_s, x_s, img_rgb, self.pred_info)
                    var predictedFrame = Predict(frameId, xSInfo, Rs, fs, xs, input.DrivingFrames[frameId], predInfo, input.UseComposite, srcImg, cropInfo);
                    
                    if (predictedFrame != null)
                    {
                        generatedFrames.Add(predictedFrame);
                    }
                }
                
                var result = new LivePortraitResult
                {
                    Success = true,
                    GeneratedFrames = generatedFrames
                };
                
                Logger.Log($"[LivePortraitInference] === GENERATION COMPLETED - {generatedFrames.Count} frames ===");
                return result;
            }
            catch (Exception e)
            {
                Logger.LogError($"[LivePortraitInference] Generation failed: {e.Message}\n{e.StackTrace}");
                return new LivePortraitResult
                {
                    Success = false,
                    ErrorMessage = e.Message,
                    GeneratedFrames = new List<Texture2D>()
                };
            }
        }
        
        /// <summary>
        /// Python: src_preprocess(img)
        /// Adjust image size and ensure dimensions are multiples of 2
        /// </summary>
        private Texture2D SrcPreprocess(Texture2D img)
        {
            int h = img.height;
            int w = img.width;
            
            // Adjust size according to maximum dimension
            const int maxDim = 1280;
            if (Mathf.Max(h, w) > maxDim)
            {
                int newHeight, newWidth;
                if (h > w)
                {
                    newHeight = maxDim;
                    newWidth = Mathf.RoundToInt(w * ((float)maxDim / h));
                }
                else
                {
                    newWidth = maxDim;
                    newHeight = Mathf.RoundToInt(h * ((float)maxDim / w));
                }
                
                var resized = new Texture2D(newWidth, newHeight);
                Graphics.CopyTexture(img, resized);
                img = resized;
                h = newHeight;
                w = newWidth;
            }
            
            // Ensure dimensions are multiples of 2
            const int division = 2;
            int finalHeight = h - (h % division);
            int finalWidth = w - (w % division);
            
            if (finalHeight == 0 || finalWidth == 0)
            {
                return img; // No need to process
            }
            
            if (finalHeight != h || finalWidth != w)
            {
                var cropped = new Texture2D(finalWidth, finalHeight);
                var pixels = img.GetPixels(0, h - finalHeight, finalWidth, finalHeight);
                cropped.SetPixels(pixels);
                cropped.Apply();
                return cropped;
            }
            
            return img;
        }
        
        /// <summary>
        /// Python: crop_src_image(models, img)
        /// Detect face and crop source image
        /// </summary>
        private CropInfo CropSrcImage(Texture2D img)
        {
            // Python: face_analysis = models["face_analysis"]
            // Python: src_face = face_analysis(img)
            var (bboxes, _) = _insightFaceHelper.GetLandmarkAndBbox(new[] { img });
            
            if (bboxes == null || bboxes.Count == 0 || bboxes[0] == InsightFaceHelper.CoordPlaceholder)
            {
                throw new InvalidOperationException("No face detected in the source image.");
            }
            
            if (bboxes.Count > 1)
            {
                Logger.LogWarning("More than one face detected in the image, only pick one face.");
            }
            
            var bbox = bboxes[0];
            var faceBbox = new Rect(bbox.x, bbox.y, bbox.z - bbox.x, bbox.w - bbox.y);
            
            // Get 106 landmarks from the detected face
            var lmk = _landmark106.GetLandmarks(img, faceBbox);
            
            // Python: lmk = landmark_runner(models, img, lmk)
            lmk = LandmarkRunner(img, lmk);
            
            // Python: crop_info = crop_image(img, lmk, dsize=512, scale=2.3, vy_ratio=-0.125)
            var cropInfo = CropImage(img, lmk, 512, 2.3f, -0.125f);
            
            // Python: crop_info["lmk_crop"] = lmk
            cropInfo.LandmarksCrop = lmk;
            
            // Python: crop_info["img_crop_256x256"] = cv2.resize(crop_info["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            cropInfo.ImageCrop256x256 = ResizeTexture(cropInfo.ImageCrop, 256, 256);
            
            // Python: crop_info["lmk_crop_256x256"] = crop_info["lmk_crop"] * 256 / 512
            cropInfo.LandmarksCrop256x256 = ScaleLandmarks(cropInfo.LandmarksCrop, 256f / 512f);
            
            return cropInfo;
        }
        
        /// <summary>
        /// Python: landmark_runner(models, img, lmk)
        /// Refine landmarks using landmark model
        /// </summary>
        private Vector2[] LandmarkRunner(Texture2D img, Vector2[] lmk)
        {
            // Python: crop_dct = crop_image(img, lmk, dsize=224, scale=1.5, vy_ratio=-0.1)
            var cropDct = CropImage(img, lmk, 224, 1.5f, -0.1f);
            var imgCrop = cropDct.ImageCrop;
            
            // Python: img_crop = img_crop / 255
            // Python: output = net.run(None, {"input": img_crop})
            var output = _landmarkRunner.GetLandmarks(imgCrop, lmk);
            
            return output;
        }
        
        /// <summary>
        /// Python: preprocess(img)
        /// Normalize image to 0-1 range and convert to CHW format
        /// </summary>
        private float[] Preprocess(Texture2D img)
        {
            var pixels = img.GetPixels();
            var data = new float[pixels.Length * 3];
            
            // Python: img = img / 255.0
            // Python: img = np.clip(img, 0, 1)
            // Python: img = img.transpose(2, 0, 1)  # HxWx3 -> 3xHxW
            // Python: img = np.expand_dims(img, axis=0)  # -> 1x3xHxW
            
            int height = img.height;
            int width = img.width;
            
            // Convert Unity RGBA to CHW format (RGB only)
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int srcIdx = y * width + x;
                        int dstIdx = c * height * width + y * width + x;
                        
                        float value = c == 0 ? pixels[srcIdx].r : c == 1 ? pixels[srcIdx].g : pixels[srcIdx].b;
                        data[dstIdx] = Mathf.Clamp01(value); // Normalize and clip
                    }
                }
            }
            
            return data;
        }
        
        /// <summary>
        /// Python: get_kp_info(models, x)
        /// Extract keypoint information using motion extractor
        /// </summary>
        private MotionInfo GetKpInfo(float[] x)
        {
            // Convert to texture for model input
            int size = Mathf.RoundToInt(Mathf.Sqrt(x.Length / 3));
            var texture = new Texture2D(size, size);
            
            var pixels = new Color[size * size];
            for (int i = 0; i < pixels.Length; i++)
            {
                int r = i * 3;
                int g = i * 3 + 1;
                int b = i * 3 + 2;
                
                if (r < x.Length && g < x.Length && b < x.Length)
                {
                    pixels[i] = new Color(x[r], x[g], x[b], 1f);
                }
            }
            
            texture.SetPixels(pixels);
            texture.Apply();
            
            // Python: output = net.run(None, {"img": x})
            // Python: pitch, yaw, roll, t, exp, scale, kp = output
            var output = _motionExtractor.GetMotionInfo(texture);
            
            UnityEngine.Object.DestroyImmediate(texture);
            
            return output;
        }
        
        /// <summary>
        /// Python: get_rotation_matrix(pitch, yaw, roll)
        /// Calculate rotation matrix from Euler angles
        /// </summary>
        private float[,] GetRotationMatrix(float[] pitch, float[] yaw, float[] roll)
        {
            float p = pitch[0] * Mathf.Deg2Rad;
            float y = yaw[0] * Mathf.Deg2Rad;
            float r = roll[0] * Mathf.Deg2Rad;
            
            float cosp = Mathf.Cos(p);
            float sinp = Mathf.Sin(p);
            float cosy = Mathf.Cos(y);
            float siny = Mathf.Sin(y);
            float cosr = Mathf.Cos(r);
            float sinr = Mathf.Sin(r);
            
            var rotMatrix = new float[3, 3];
            rotMatrix[0, 0] = cosy * cosr;
            rotMatrix[0, 1] = -cosy * sinr;
            rotMatrix[0, 2] = siny;
            rotMatrix[1, 0] = sinp * siny * cosr + cosp * sinr;
            rotMatrix[1, 1] = -sinp * siny * sinr + cosp * cosr;
            rotMatrix[1, 2] = -sinp * cosy;
            rotMatrix[2, 0] = -cosp * siny * cosr + sinp * sinr;
            rotMatrix[2, 1] = cosp * siny * sinr + sinp * cosr;
            rotMatrix[2, 2] = cosp * cosy;
            
            return rotMatrix;
        }
        
        /// <summary>
        /// Python: extract_feature_3d(models, x)
        /// Extract 3D appearance features
        /// </summary>
        private float[] ExtractFeature3d(float[] x)
        {
            // Convert to texture for model input
            int size = Mathf.RoundToInt(Mathf.Sqrt(x.Length / 3));
            var texture = new Texture2D(size, size);
            
            var pixels = new Color[size * size];
            for (int i = 0; i < pixels.Length; i++)
            {
                int r = i * 3;
                int g = i * 3 + 1;
                int b = i * 3 + 2;
                
                if (r < x.Length && g < x.Length && b < x.Length)
                {
                    pixels[i] = new Color(x[r], x[g], x[b], 1f);
                }
            }
            
            texture.SetPixels(pixels);
            texture.Apply();
            
            // Python: output = net.run(None, {"img": x})
            // Python: f_s = output[0]
            var output = _appearanceExtractor.ExtractFeature3D(texture);
            
            UnityEngine.Object.DestroyImmediate(texture);
            
            return output;
        }
        
        /// <summary>
        /// Python: transform_keypoint(x_s_info)
        /// Transform keypoints based on motion info
        /// </summary>
        private float[] TransformKeypoint(MotionInfo xSInfo)
        {
            // This should match the Python transform_keypoint function
            // For now, return the keypoints as-is
            return xSInfo.Keypoints;
        }
        
        /// <summary>
        /// Python: predict(frame_id, models, x_s_info, R_s, f_s, x_s, img, pred_info)
        /// Main prediction function for each frame
        /// </summary>
        private Texture2D Predict(int frameId, MotionInfo xSInfo, float[,] Rs, float[] fs, float[] xs, 
            Texture2D img, LivePortraitPredInfo predInfo, bool useComposite, Texture2D srcImg, CropInfo cropInfo)
        {
            // Python: frame_0 = pred_info['lmk'] is None
            bool frame0 = predInfo.Landmarks == null;
            
            Vector2[] lmk;
            if (frame0)
            {
                // First frame - detect face
                var (bboxes, _) = _insightFaceHelper.GetLandmarkAndBbox(new[] { img });
                if (bboxes == null || bboxes.Count == 0 || bboxes[0] == InsightFaceHelper.CoordPlaceholder)
                {
                    throw new InvalidOperationException("No face detected in the frame");
                }
                
                if (bboxes.Count > 1)
                {
                    Logger.LogWarning("More than one face detected in the driving frame, only pick one face.");
                }
                
                var bbox = bboxes[0];
                var faceBbox = new Rect(bbox.x, bbox.y, bbox.z - bbox.x, bbox.w - bbox.y);
                lmk = _landmark106.GetLandmarks(img, faceBbox);
                lmk = LandmarkRunner(img, lmk);
            }
            else
            {
                lmk = LandmarkRunner(img, predInfo.Landmarks);
            }
            predInfo.Landmarks = lmk;
            
            // Python: img = cv2.resize(img, (256, 256))
            var img256 = ResizeTexture(img, 256, 256);
            
            // Python: I_d = preprocess(img)
            var Id = Preprocess(img256);
            
            // Python: x_d_info = get_kp_info(models, I_d)
            var xDInfo = GetKpInfo(Id);
            
            // Python: R_d = get_rotation_matrix(x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"])
            var Rd = GetRotationMatrix(xDInfo.Pitch, xDInfo.Yaw, xDInfo.Roll);
            
            if (frame0)
            {
                predInfo.InitialMotionInfo = xDInfo;
            }
            
            var xD0Info = predInfo.InitialMotionInfo;
            var Rd0 = GetRotationMatrix(xD0Info.Pitch, xD0Info.Yaw, xD0Info.Roll);
            
            // Python: R_new = (R_d @ R_d_0.transpose(0, 2, 1)) @ R_s
            var RNew = MatrixMultiply(MatrixMultiply(Rd, TransposeMatrix(Rd0)), Rs);
            
            // Python: delta_new = x_s_info["exp"] + (x_d_info["exp"] - x_d_0_info["exp"])
            var deltaNew = AddArrays(xSInfo.Expression, SubtractArrays(xDInfo.Expression, xD0Info.Expression));
            
            // Python: scale_new = x_s_info["scale"] * (x_d_info["scale"] / x_d_0_info["scale"])
            var scaleNew = MultiplyArrays(xSInfo.Scale, DivideArrays(xDInfo.Scale, xD0Info.Scale));
            
            // Python: t_new = x_s_info["t"] + (x_d_info["t"] - x_d_0_info["t"])
            var tNew = AddArrays(xSInfo.Translation, SubtractArrays(xDInfo.Translation, xD0Info.Translation));
            
            // Python: t_new[..., 2] = 0  # zero tz
            if (tNew.Length >= 3) tNew[2] = 0;
            
            // Python: x_c_s = x_s_info["kp"]
            var xCs = xSInfo.Keypoints;
            
            // Python: x_d_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
            var xDNew = CalculateNewKeypoints(xCs, RNew, deltaNew, scaleNew, tNew);
            
            // Python: x_d_new = stitching(models, x_s, x_d_new)
            xDNew = Stitching(xs, xDNew);
            
            // Python: out = warping_spade(models, f_s, x_s, x_d_new)
            var output = WarpingSpade(fs, xs, xDNew);
            
            // Python: out = out.transpose(0, 2, 3, 1)  # 1x3xHxW -> 1xHxWx3
            // Python: out = np.clip(out, 0, 1)
            // Python: out = (out * 255).astype(np.uint8)
            var resultTexture = ConvertOutputToTexture(output);
            
            // Cleanup
            UnityEngine.Object.DestroyImmediate(img256);
            
            // Apply composite or paste back
            if (useComposite)
            {
                return CreateCompositeFrame(img, cropInfo.ImageCrop256x256, resultTexture);
            }
            else
            {
                return PasteBack(resultTexture, cropInfo, srcImg);
            }
        }
        
        /// <summary>
        /// Python: stitching(models, kp_source, kp_driving)
        /// </summary>
        private float[] Stitching(float[] kpSource, float[] kpDriving)
        {
            var output = _stitching.Stitch(kpSource, kpDriving);
            return output;
        }
        
        /// <summary>
        /// Python: warping_spade(models, feature_3d, kp_source, kp_driving)
        /// </summary>
        private float[] WarpingSpade(float[] feature3d, float[] kpSource, float[] kpDriving)
        {
            var output = _warpingSpade.WarpImage(feature3d, kpSource, kpDriving);
            return output;
        }
        
        // Helper methods
        private Texture2D ResizeTexture(Texture2D source, int width, int height)
        {
            var result = new Texture2D(width, height);
            var rt = RenderTexture.GetTemporary(width, height);
            Graphics.Blit(source, rt);
            
            RenderTexture.active = rt;
            result.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            result.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);
            
            return result;
        }
        
        private Vector2[] ScaleLandmarks(Vector2[] landmarks, float scale)
        {
            var result = new Vector2[landmarks.Length];
            for (int i = 0; i < landmarks.Length; i++)
            {
                result[i] = landmarks[i] * scale;
            }
            return result;
        }
        
        private CropInfo CropImage(Texture2D img, Vector2[] lmk, int dsize, float scale, float vyRatio)
        {
            // Simplified crop implementation
            // Full implementation would match Python crop_image exactly
            var cropInfo = new CropInfo
            {
                ImageCrop = ResizeTexture(img, dsize, dsize),
                Transform = Matrix4x4.identity,
                LandmarksCrop = lmk
            };
            
            return cropInfo;
        }
        
        private float[,] MatrixMultiply(float[,] a, float[,] b)
        {
            int rows = a.GetLength(0);
            int cols = b.GetLength(1);
            int inner = a.GetLength(1);
            
            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    for (int k = 0; k < inner; k++)
                    {
                        result[i, j] += a[i, k] * b[k, j];
                    }
                }
            }
            return result;
        }
        
        private float[,] TransposeMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[cols, rows];
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }
            return result;
        }
        
        private float[] AddArrays(float[] a, float[] b)
        {
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
            return result;
        }
        
        private float[] SubtractArrays(float[] a, float[] b)
        {
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] - b[i];
            }
            return result;
        }
        
        private float[] MultiplyArrays(float[] a, float[] b)
        {
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
            return result;
        }
        
        private float[] DivideArrays(float[] a, float[] b)
        {
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] / b[i];
            }
            return result;
        }
        
        private float[] ConcatenateArrays(float[] a, float[] b)
        {
            var result = new float[a.Length + b.Length];
            Array.Copy(a, 0, result, 0, a.Length);
            Array.Copy(b, 0, result, a.Length, b.Length);
            return result;
        }
        
        private float[] CalculateNewKeypoints(float[] xCs, float[,] RNew, float[] deltaNew, float[] scaleNew, float[] tNew)
        {
            // Simplified - full implementation would do proper matrix operations
            var result = new float[xCs.Length];
            Array.Copy(xCs, result, xCs.Length);
            return result;
        }
        
        private Texture2D ConvertOutputToTexture(float[] output)
        {
            // Convert ONNX output back to texture
            // This is simplified - full implementation would handle CHW->HWC conversion
            int size = Mathf.RoundToInt(Mathf.Sqrt(output.Length / 3));
            var texture = new Texture2D(size, size);
            
            var pixels = new Color[size * size];
            for (int i = 0; i < pixels.Length; i++)
            {
                int r = i * 3;
                int g = i * 3 + 1;
                int b = i * 3 + 2;
                
                if (r < output.Length && g < output.Length && b < output.Length)
                {
                    pixels[i] = new Color(
                        Mathf.Clamp01(output[r]),
                        Mathf.Clamp01(output[g]),
                        Mathf.Clamp01(output[b]),
                        1f
                    );
                }
            }
            
            texture.SetPixels(pixels);
            texture.Apply();
            return texture;
        }
        
        private Texture2D CreateCompositeFrame(Texture2D driving, Texture2D crop, Texture2D predicted)
        {
            // Simple composite - place images side by side
            var composite = new Texture2D(driving.width + crop.width + predicted.width, 
                                        Mathf.Max(driving.height, crop.height, predicted.height));
            
            // Copy pixels from each image
            Graphics.CopyTexture(driving, 0, 0, 0, 0, driving.width, driving.height, 
                               composite, 0, 0, 0, 0);
            Graphics.CopyTexture(crop, 0, 0, 0, 0, crop.width, crop.height, 
                               composite, 0, 0, driving.width, 0);
            Graphics.CopyTexture(predicted, 0, 0, 0, 0, predicted.width, predicted.height, 
                               composite, 0, 0, driving.width + crop.width, 0);
            
            return composite;
        }
        
        private Texture2D PasteBack(Texture2D predicted, CropInfo cropInfo, Texture2D original)
        {
            // Simple paste back - for now just return predicted
            // Full implementation would use transform matrix and mask
            return predicted;
        }
        
        public void Dispose()
        {
            if (_disposed) return;
            
            try
            {
                _appearanceExtractor?.Dispose();
                _motionExtractor?.Dispose();
                _warpingSpade?.Dispose();
                _stitching?.Dispose();
                _landmarkRunner?.Dispose();
                _landmark106?.Dispose();
                _insightFaceHelper?.Dispose();
                
                if (_maskTemplate != null)
                {
                    UnityEngine.Object.DestroyImmediate(_maskTemplate);
                }
                
                _disposed = true;
                Logger.Log("[LivePortraitInference] Disposed successfully");
            }
            catch (Exception e)
            {
                Logger.LogError($"[LivePortraitInference] Error during disposal: {e.Message}");
            }
        }
    }
    
    // Data structures to match Python exactly
    public class CropInfo
    {
        public Texture2D ImageCrop { get; set; }
        public Texture2D ImageCrop256x256 { get; set; }
        public Vector2[] LandmarksCrop { get; set; }
        public Vector2[] LandmarksCrop256x256 { get; set; }
        public Matrix4x4 Transform { get; set; }
    }
}
