using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MuseTalk.Models
{
    using Utils;

    /// <summary>
    /// LivePortrait Appearance Feature Extractor model
    /// Extracts 3D appearance features from face images
    /// </summary>
    public class AppearanceFeatureExtractorModel : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        private readonly InferenceSession _session;
        private bool _disposed = false;
        
        public bool IsInitialized { get; private set; }
        
        public AppearanceFeatureExtractorModel(MuseTalkConfig config)
        {
            try
            {
                _session = ModelUtils.LoadModel(config, "appearance_feature_extractor");
                IsInitialized = true;
                Logger.Log("[AppearanceFeatureExtractorModel] Initialized successfully");
            }
            catch (Exception e)
            {
                Logger.LogError($"[AppearanceFeatureExtractorModel] Failed to initialize: {e.Message}");
                IsInitialized = false;
            }
        }
        
        /// <summary>
        /// Extract feature_3d from image, matching Python extract_feature_3d exactly
        /// </summary>
        public float[] ExtractFeature3D(Texture2D image)
        {
            if (!IsInitialized) return null;
            
            try
            {
                // Preprocess image to tensor [1, 3, 256, 256]
                var inputTensor = PreprocessImage(image);
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("img", inputTensor)
                };
                
                using var results = _session.Run(inputs);
                var output = results.First().AsTensor<float>();
                return output.ToArray();
            }
            catch (Exception e)
            {
                Logger.LogError($"[AppearanceFeatureExtractorModel] Feature extraction failed: {e.Message}");
                return null;
            }
        }
        
        private DenseTensor<float> PreprocessImage(Texture2D texture)
        {
            // Resize to 256x256 if needed
            var resizedTexture = TextureUtils.ResizeTexture(texture, 256, 256);
            
            // Convert to tensor with normalization (matching Python preprocessing)
            var tensorData = new float[1 * 3 * 256 * 256];
            var pixels = resizedTexture.GetPixels();
            
            int idx = 0;
            // Convert RGB to tensor format [1, 3, H, W] with normalization
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 256; h++)
                {
                    for (int w = 0; w < 256; w++)
                    {
                        float pixelValue = c == 0 ? pixels[h * 256 + w].r : 
                                          c == 1 ? pixels[h * 256 + w].g : 
                                                   pixels[h * 256 + w].b;
                        // Normalize to [0, 1] then to [-1, 1]
                        tensorData[idx++] = (pixelValue * 2.0f) - 1.0f;
                    }
                }
            }
            
            if (resizedTexture != texture)
                UnityEngine.Object.DestroyImmediate(resizedTexture);
            
            return new DenseTensor<float>(tensorData, new[] { 1, 3, 256, 256 });
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                Logger.Log("[AppearanceFeatureExtractorModel] Disposed");
            }
        }
    }

    /// <summary>
    /// LivePortrait Motion Extractor model  
    /// Extracts motion parameters (pitch, yaw, roll, translation, expression, scale, keypoints)
    /// </summary>
    public class MotionExtractorModel : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        private readonly InferenceSession _session;
        private bool _disposed = false;
        
        public bool IsInitialized { get; private set; }
        
        public MotionExtractorModel(MuseTalkConfig config)
        {
            try
            {
                _session = ModelUtils.LoadModel(config, "motion_extractor");
                IsInitialized = true;
                Logger.Log("[MotionExtractorModel] Initialized successfully");
            }
            catch (Exception e)
            {
                Logger.LogError($"[MotionExtractorModel] Failed to initialize: {e.Message}");
                IsInitialized = false;
            }
        }
        
        /// <summary>
        /// Extract motion parameters, matching Python get_kp_info exactly
        /// </summary>
        public MotionInfo GetMotionInfo(Texture2D image)
        {
            if (!IsInitialized) return null;
            
            try
            {
                var inputTensor = PreprocessImage(image);
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("img", inputTensor)
                };
                
                using var results = _session.Run(inputs);
                
                // Extract outputs matching Python: pitch, yaw, roll, t, exp, scale, kp
                var outputs = results.ToArray();
                var pitch = outputs[0].AsTensor<float>().ToArray();
                var yaw = outputs[1].AsTensor<float>().ToArray();
                var roll = outputs[2].AsTensor<float>().ToArray();
                var t = outputs[3].AsTensor<float>().ToArray();
                var exp = outputs[4].AsTensor<float>().ToArray();
                var scale = outputs[5].AsTensor<float>().ToArray();
                var kp = outputs[6].AsTensor<float>().ToArray();
                
                // Process angles using softmax (matching Python)
                var processedPitch = ProcessAngleSoftmax(pitch);
                var processedYaw = ProcessAngleSoftmax(yaw);
                var processedRoll = ProcessAngleSoftmax(roll);
                
                return new MotionInfo
                {
                    Pitch = processedPitch,
                    Yaw = processedYaw,
                    Roll = processedRoll,
                    Translation = t,
                    Expression = exp,
                    Scale = scale,
                    Keypoints = kp
                };
            }
            catch (Exception e)
            {
                Logger.LogError($"[MotionExtractorModel] Motion extraction failed: {e.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Process angle predictions using softmax, matching Python exactly
        /// </summary>
        private float[] ProcessAngleSoftmax(float[] angleLogits)
        {
            // Apply softmax
            var softmaxValues = Softmax(angleLogits);
            
            // Convert to degrees: degree = sum(pred * arange(66)) * 3 - 97.5
            float degree = 0f;
            for (int i = 0; i < softmaxValues.Length; i++)
            {
                degree += softmaxValues[i] * i;
            }
            degree = degree * 3f - 97.5f;
            
            return new float[] { degree };
        }
        
        private float[] Softmax(float[] logits)
        {
            var maxVal = logits.Max();
            var exps = new float[logits.Length];
            float sum = 0f;
            
            for (int i = 0; i < logits.Length; i++)
            {
                exps[i] = Mathf.Exp(logits[i] - maxVal);
                sum += exps[i];
            }
            
            for (int i = 0; i < exps.Length; i++)
            {
                exps[i] /= sum;
            }
            
            return exps;
        }
        
        private DenseTensor<float> PreprocessImage(Texture2D texture)
        {
            // Same preprocessing as AppearanceFeatureExtractor
            var resizedTexture = TextureUtils.ResizeTexture(texture, 256, 256);
            
            var tensorData = new float[1 * 3 * 256 * 256];
            var pixels = resizedTexture.GetPixels();
            
            int idx = 0;
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 256; h++)
                {
                    for (int w = 0; w < 256; w++)
                    {
                        float pixelValue = c == 0 ? pixels[h * 256 + w].r : 
                                          c == 1 ? pixels[h * 256 + w].g : 
                                                   pixels[h * 256 + w].b;
                        tensorData[idx++] = (pixelValue * 2.0f) - 1.0f;
                    }
                }
            }
            
            if (resizedTexture != texture)
                UnityEngine.Object.DestroyImmediate(resizedTexture);
            
            return new DenseTensor<float>(tensorData, new[] { 1, 3, 256, 256 });
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                Logger.Log("[MotionExtractorModel] Disposed");
            }
        }
    }

    /// <summary>
    /// LivePortrait Warping SPADE model
    /// Performs neural warping based on keypoints and appearance features
    /// </summary>
    public class WarpingSPADEModel : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        private readonly InferenceSession _session;
        private bool _disposed = false;
        
        public bool IsInitialized { get; private set; }
        
        public WarpingSPADEModel(MuseTalkConfig config)
        {
            try
            {
                _session = ModelUtils.LoadModel(config, "warping_spade");
                IsInitialized = true;
                
                // Log ONNX Runtime version and model info
                Logger.Log($"[WarpingSPADEModel] Initialized successfully");
                Logger.Log($"[WarpingSPADEModel] ONNX Runtime version: {Microsoft.ML.OnnxRuntime.OrtEnv.Instance().GetVersionString()}");
                
                // Log model input/output metadata
                var inputMetadata = _session.InputMetadata;
                var outputMetadata = _session.OutputMetadata;
                
                Logger.Log($"[WarpingSPADEModel] Model inputs:");
                foreach (var input in inputMetadata)
                {
                    Logger.Log($"[WarpingSPADEModel]   {input.Key}: {string.Join("x", input.Value.Dimensions)} ({input.Value.ElementType})");
                }
                
                Logger.Log($"[WarpingSPADEModel] Model outputs:");
                foreach (var output in outputMetadata)
                {
                    Logger.Log($"[WarpingSPADEModel]   {output.Key}: {string.Join("x", output.Value.Dimensions)} ({output.Value.ElementType})");
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[WarpingSPADEModel] Failed to initialize: {e.Message}");
                IsInitialized = false;
            }
        }
        
        /// <summary>
        /// Perform warping, matching Python warping_spade exactly
        /// </summary>
        public float[] WarpImage(float[] feature3D, float[] kpSource, float[] kpDriving)
        {
            if (!IsInitialized) return null;
            
            try
            {
                // Debug input shapes
                var feature3DShape = GetFeature3DShape(feature3D);
                var kpSourceShape = GetKeypointsShape(kpSource);
                var kpDrivingShape = GetKeypointsShape(kpDriving);
                
                Logger.Log($"[WarpingSPADEModel] Input shapes:");
                Logger.Log($"[WarpingSPADEModel]   feature3D: {feature3D.Length} -> [{string.Join(",", feature3DShape.Select(x => x.ToString()))}]");
                Logger.Log($"[WarpingSPADEModel]   kpSource: {kpSource.Length} -> [{string.Join(",", kpSourceShape.Select(x => x.ToString()))}]");
                Logger.Log($"[WarpingSPADEModel]   kpDriving: {kpDriving.Length} -> [{string.Join(",", kpDrivingShape.Select(x => x.ToString()))}]");
                
                // Verify shapes match expected sizes
                int expectedFeature3DSize = feature3DShape.Aggregate(1, (a, b) => a * b);
                int expectedKpSourceSize = kpSourceShape.Aggregate(1, (a, b) => a * b);
                int expectedKpDrivingSize = kpDrivingShape.Aggregate(1, (a, b) => a * b);
                
                if (feature3D.Length != expectedFeature3DSize)
                {
                    Logger.LogError($"[WarpingSPADEModel] Feature3D size mismatch: got {feature3D.Length}, expected {expectedFeature3DSize}");
                    return null;
                }
                
                if (kpSource.Length != expectedKpSourceSize)
                {
                    Logger.LogError($"[WarpingSPADEModel] KpSource size mismatch: got {kpSource.Length}, expected {expectedKpSourceSize}");
                    return null;
                }
                
                if (kpDriving.Length != expectedKpDrivingSize)
                {
                    Logger.LogError($"[WarpingSPADEModel] KpDriving size mismatch: got {kpDriving.Length}, expected {expectedKpDrivingSize}");
                    return null;
                }
                
                // Convert arrays to tensors with proper shapes
                var feature3DTensor = new DenseTensor<float>(feature3D, feature3DShape);
                var kpSourceTensor = new DenseTensor<float>(kpSource, kpSourceShape);
                var kpDrivingTensor = new DenseTensor<float>(kpDriving, kpDrivingShape);
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("feature_3d", feature3DTensor),
                    NamedOnnxValue.CreateFromTensor("kp_source", kpSourceTensor),
                    NamedOnnxValue.CreateFromTensor("kp_driving", kpDrivingTensor)
                };
                
                // Debug input tensor details
                Logger.Log($"[WarpingSPADEModel] Input tensor details:");
                Logger.Log($"[WarpingSPADEModel]   feature3D tensor: {feature3DTensor.Length} elements, dtype: {feature3DTensor.GetType()}");
                Logger.Log($"[WarpingSPADEModel]   kpSource tensor: {kpSourceTensor.Length} elements, dtype: {kpSourceTensor.GetType()}");
                Logger.Log($"[WarpingSPADEModel]   kpDriving tensor: {kpDrivingTensor.Length} elements, dtype: {kpDrivingTensor.GetType()}");
                
                // Check for NaN or infinite values in inputs
                bool hasNaN = feature3D.Any(x => float.IsNaN(x) || float.IsInfinity(x)) ||
                             kpSource.Any(x => float.IsNaN(x) || float.IsInfinity(x)) ||
                             kpDriving.Any(x => float.IsNaN(x) || float.IsInfinity(x));
                Logger.Log($"[WarpingSPADEModel] Input contains NaN/Inf: {hasNaN}");
                
                // Check input value ranges
                Logger.Log($"[WarpingSPADEModel] Input ranges:");
                Logger.Log($"[WarpingSPADEModel]   feature3D: [{feature3D.Min():F6}, {feature3D.Max():F6}]");
                Logger.Log($"[WarpingSPADEModel]   kpSource: [{kpSource.Min():F6}, {kpSource.Max():F6}]");
                Logger.Log($"[WarpingSPADEModel]   kpDriving: [{kpDriving.Min():F6}, {kpDriving.Max():F6}]");
                
                Logger.Log($"[WarpingSPADEModel] Running ONNX inference...");
                using var results = _session.Run(inputs);
                var output = results.First().AsTensor<float>();
                var outputArray = output.ToArray();
                
                Logger.Log($"[WarpingSPADEModel] ONNX output: {outputArray.Length} elements, shape: [{string.Join(",", output.Dimensions.ToArray().Select(x => x.ToString()))}]");
                
                // Check output for issues
                bool outputHasNaN = outputArray.Any(x => float.IsNaN(x) || float.IsInfinity(x));
                Logger.Log($"[WarpingSPADEModel] Output contains NaN/Inf: {outputHasNaN}");
                Logger.Log($"[WarpingSPADEModel] Output range: [{outputArray.Min():F6}, {outputArray.Max():F6}]");
                
                // Check if output is all zeros
                bool allZeros = outputArray.All(x => Math.Abs(x) < 1e-10f);
                Logger.Log($"[WarpingSPADEModel] Output is all zeros: {allZeros}");
                
                return outputArray;
            }
            catch (Exception e)
            {
                Logger.LogError($"[WarpingSPADEModel] Warping failed: {e.Message}");
                Logger.LogError($"[WarpingSPADEModel] Stack trace: {e.StackTrace}");
                return null;
            }
        }
        
        private int[] GetFeature3DShape(float[] feature3D)
        {
            // Typical shape is [1, 32, 16, 64, 64] for LivePortrait
            return new[] { 1, 32, 16, 64, 64 };
        }
        
        private int[] GetKeypointsShape(float[] keypoints)
        {
            // Typical shape is [1, N, 3] where N is number of keypoints
            int numKeypoints = keypoints.Length / 3;
            return new[] { 1, numKeypoints, 3 };
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                Logger.Log("[WarpingSPADEModel] Disposed");
            }
        }
    }

    /// <summary>
    /// LivePortrait Stitching model
    /// Performs stitching between source and driving keypoints
    /// </summary>
    public class StitchingModel : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        private readonly InferenceSession _session;
        private bool _disposed = false;
        
        public bool IsInitialized { get; private set; }
        
        public StitchingModel(MuseTalkConfig config)
        {
            try
            {
                _session = ModelUtils.LoadModel(config, "stitching");
                IsInitialized = true;
                Logger.Log("[StitchingModel] Initialized successfully");
            }
            catch (Exception e)
            {
                Logger.LogError($"[StitchingModel] Failed to initialize: {e.Message}");
                IsInitialized = false;
            }
        }
        
        /// <summary>
        /// Perform stitching, matching Python stitching exactly
        /// </summary>
        public float[] Stitch(float[] kpSource, float[] kpDriving)
        {
            if (!IsInitialized) return kpDriving; // Return original if model fails
            
            try
            {
                // Flatten and concatenate keypoints as in Python
                var concatenated = new float[kpSource.Length + kpDriving.Length];
                Array.Copy(kpSource, 0, concatenated, 0, kpSource.Length);
                Array.Copy(kpDriving, 0, concatenated, kpSource.Length, kpDriving.Length);
                
                var inputTensor = new DenseTensor<float>(concatenated, new[] { 1, concatenated.Length });
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input", inputTensor)
                };
                
                using var results = _session.Run(inputs);
                var delta = results.First().AsTensor<float>().ToArray();
                
                // Apply delta to driving keypoints
                return ApplyStitchingDelta(kpDriving, delta);
            }
            catch (Exception e)
            {
                Logger.LogError($"[StitchingModel] Stitching failed: {e.Message}");
                return kpDriving; // Return original keypoints
            }
        }
        
        private float[] ApplyStitchingDelta(float[] kpDriving, float[] delta)
        {
            // Apply delta according to Python logic
            var result = new float[kpDriving.Length];
            Array.Copy(kpDriving, result, kpDriving.Length);
            
            int numKeypoints = kpDriving.Length / 3;
            
            // Apply expression delta (first part of delta)
            for (int i = 0; i < numKeypoints * 3; i++)
            {
                if (i < delta.Length)
                    result[i] += delta[i];
            }
            
            // Apply translation delta (last 2 elements of delta)
            if (delta.Length >= numKeypoints * 3 + 2)
            {
                float deltaX = delta[numKeypoints * 3];
                float deltaY = delta[numKeypoints * 3 + 1];
                
                for (int i = 0; i < numKeypoints; i++)
                {
                    result[i * 3] += deltaX;     // x coordinate
                    result[i * 3 + 1] += deltaY; // y coordinate
                }
            }
            
            return result;
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                Logger.Log("[StitchingModel] Disposed");
            }
        }
    }

    /// <summary>
    /// LivePortrait Landmark model (106 landmarks)
    /// Enhanced landmark detection for LivePortrait pipeline
    /// </summary>
    public class LivePortraitLandmarkModel : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        private readonly InferenceSession _session;
        private bool _disposed = false;
        
        public bool IsInitialized { get; private set; }
        
        public LivePortraitLandmarkModel(MuseTalkConfig config)
        {
            try
            {
                _session = ModelUtils.LoadModel(config, "landmark");
                IsInitialized = true;
                Logger.Log("[LivePortraitLandmarkModel] Initialized successfully");
            }
            catch (Exception e)
            {
                Logger.LogError($"[LivePortraitLandmarkModel] Failed to initialize: {e.Message}");
                IsInitialized = false;
            }
        }
        
        /// <summary>
        /// Extract 203-point landmarks, matching Python landmark_runner exactly
        /// </summary>
        public Vector2[] GetLandmarks(Texture2D image, Vector2[] initialLandmarks)
        {
            if (!IsInitialized) return new Vector2[0];
            
            try
            {
                // Crop image based on initial landmarks
                var cropInfo = CropImageForLandmarks(image, initialLandmarks);
                
                // Preprocess cropped image
                var inputTensor = PreprocessImageForLandmarks(cropInfo.croppedImage);
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input", inputTensor)
                };
                
                using var results = _session.Run(inputs);
                var output = results.ToArray()[2]; // Third output contains the landmarks
                var landmarks = output.AsTensor<float>().ToArray();
                
                // Convert to Vector2 array and scale to original coordinates
                return ProcessLandmarkOutput(landmarks, cropInfo.transform);
            }
            catch (Exception e)
            {
                Logger.LogError($"[LivePortraitLandmarkModel] Landmark extraction failed: {e.Message}");
                return new Vector2[0];
            }
        }
        
        private (Texture2D croppedImage, Matrix4x4 transform) CropImageForLandmarks(Texture2D image, Vector2[] landmarks)
        {
            // Use landmarks to create crop region (matching Python crop_image logic)
            var cropRect = CalculateCropRect(landmarks, image.width, image.height);
            var croppedTexture = TextureUtils.CropTexture(image, cropRect);
            var resizedTexture = TextureUtils.ResizeTexture(croppedTexture, 224, 224);
            
            // Calculate transform matrix for coordinate mapping
            var transform = Matrix4x4.identity;
            transform.m00 = cropRect.width / 224f;
            transform.m11 = cropRect.height / 224f;
            transform.m03 = cropRect.x;
            transform.m13 = cropRect.y;
            
            UnityEngine.Object.DestroyImmediate(croppedTexture);
            return (resizedTexture, transform);
        }
        
        private Rect CalculateCropRect(Vector2[] landmarks, int imageWidth, int imageHeight)
        {
            // Calculate bounding box of landmarks with padding
            float minX = float.MaxValue, minY = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue;
            
            foreach (var landmark in landmarks)
            {
                minX = Mathf.Min(minX, landmark.x);
                minY = Mathf.Min(minY, landmark.y);
                maxX = Mathf.Max(maxX, landmark.x);
                maxY = Mathf.Max(maxY, landmark.y);
            }
            
            // Add padding (matching Python scale=1.5, vy_ratio=-0.1)
            float width = maxX - minX;
            float height = maxY - minY;
            float scale = 1.5f;
            float expandedWidth = width * scale;
            float expandedHeight = height * scale;
            
            float centerX = (minX + maxX) * 0.5f;
            float centerY = (minY + maxY) * 0.5f + height * -0.1f; // vy_ratio adjustment
            
            float x = Mathf.Max(0, centerX - expandedWidth * 0.5f);
            float y = Mathf.Max(0, centerY - expandedHeight * 0.5f);
            float w = Mathf.Min(imageWidth - x, expandedWidth);
            float h = Mathf.Min(imageHeight - y, expandedHeight);
            
            return new Rect(x, y, w, h);
        }
        
        private DenseTensor<float> PreprocessImageForLandmarks(Texture2D texture)
        {
            var tensorData = new float[1 * 3 * 224 * 224];
            var pixels = texture.GetPixels();
            
            int idx = 0;
            // Normalize to [0, 1] range
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 224; h++)
                {
                    for (int w = 0; w < 224; w++)
                    {
                        float pixelValue = c == 0 ? pixels[h * 224 + w].r : 
                                          c == 1 ? pixels[h * 224 + w].g : 
                                                   pixels[h * 224 + w].b;
                        tensorData[idx++] = pixelValue;
                    }
                }
            }
            
            return new DenseTensor<float>(tensorData, new[] { 1, 3, 224, 224 });
        }
        
        private Vector2[] ProcessLandmarkOutput(float[] landmarks, Matrix4x4 transform)
        {
            int numLandmarks = landmarks.Length / 2;
            var result = new Vector2[numLandmarks];
            
            for (int i = 0; i < numLandmarks; i++)
            {
                float x = landmarks[i * 2] * 224; // Scale to 224x224
                float y = landmarks[i * 2 + 1] * 224;
                
                // Transform back to original image coordinates
                Vector3 transformedPoint = transform.MultiplyPoint3x4(new Vector3(x, y, 0));
                result[i] = new Vector2(transformedPoint.x, transformedPoint.y);
            }
            
            return result;
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                Logger.Log("[LivePortraitLandmarkModel] Disposed");
            }
        }
    }

    /// <summary>
    /// 2D 106-point landmark detection model
    /// Used for initial face landmark detection in LivePortrait pipeline
    /// </summary>
    public class Landmark106Model : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        private readonly InferenceSession _session;
        private bool _disposed = false;
        
        public bool IsInitialized { get; private set; }
        
        public Landmark106Model(MuseTalkConfig config)
        {
            try
            {
                _session = ModelUtils.LoadModel(config, "2d106det");
                IsInitialized = true;
                Logger.Log("[Landmark106Model] Initialized successfully");
            }
            catch (Exception e)
            {
                Logger.LogError($"[Landmark106Model] Failed to initialize: {e.Message}");
                IsInitialized = false;
            }
        }
        
        /// <summary>
        /// Get 106 landmarks from face region, matching Python get_landmark exactly
        /// </summary>
        public Vector2[] GetLandmarks(Texture2D image, Rect faceBbox)
        {
            if (!IsInitialized) return new Vector2[0];
            
            try
            {
                // Face align crop matching Python logic
                var (alignedImage, transform) = FaceAlign(image, faceBbox);
                
                // Preprocess
                var inputTensor = PreprocessForLandmarks(alignedImage);
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("data", inputTensor)
                };
                
                using var results = _session.Run(inputs);
                var output = results.First().AsTensor<float>().ToArray();
                
                // Process landmarks
                return ProcessLandmarks106(output, transform);
            }
            catch (Exception e)
            {
                Logger.LogError($"[Landmark106Model] Landmark detection failed: {e.Message}");
                return new Vector2[0];
            }
        }
        
        private (Texture2D, Matrix4x4) FaceAlign(Texture2D image, Rect bbox)
        {
            // Calculate face alignment parameters
            float w = bbox.width;
            float h = bbox.height;
            Vector2 center = new Vector2(bbox.x + w * 0.5f, bbox.y + h * 0.5f);
            // float rotate = 0f;
            float scale = 192f / (Mathf.Max(w, h) * 1.5f);
            
            // Create aligned crop
            var alignedTexture = TextureUtils.ResizeTexture(image, 192, 192);
            
            // Calculate inverse transform matrix
            var transform = Matrix4x4.identity;
            transform.m00 = 1f / scale;
            transform.m11 = 1f / scale;
            
            return (alignedTexture, transform);
        }
        
        private DenseTensor<float> PreprocessForLandmarks(Texture2D texture)
        {
            var tensorData = new float[1 * 3 * 192 * 192];
            var pixels = texture.GetPixels();
            
            int idx = 0;
            // Convert to CHW format with proper normalization
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 192; h++)
                {
                    for (int w = 0; w < 192; w++)
                    {
                        float pixelValue = c == 0 ? pixels[h * 192 + w].r : 
                                          c == 1 ? pixels[h * 192 + w].g : 
                                                   pixels[h * 192 + w].b;
                        tensorData[idx++] = pixelValue * 255f; // [0, 255] range
                    }
                }
            }
            
            return new DenseTensor<float>(tensorData, new[] { 1, 3, 192, 192 });
        }
        
        private Vector2[] ProcessLandmarks106(float[] output, Matrix4x4 transform)
        {
            int numLandmarks = 106;
            var landmarks = new Vector2[numLandmarks];
            
            for (int i = 0; i < numLandmarks; i++)
            {
                float x = output[i * 2];
                float y = output[i * 2 + 1];
                
                // Scale and adjust coordinates
                x = (x + 1) * 96f; // 192 / 2
                y = (y + 1) * 96f;
                
                // Transform to original image space
                Vector3 transformedPoint = transform.MultiplyPoint3x4(new Vector3(x, y, 0));
                landmarks[i] = new Vector2(transformedPoint.x, transformedPoint.y);
            }
            
            return landmarks;
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                Logger.Log("[Landmark106Model] Disposed");
            }
        }
    }

    /// <summary>
    /// Data structure for motion information from MotionExtractorModel
    /// </summary>
    public class MotionInfo
    {
        public float[] Pitch { get; set; }
        public float[] Yaw { get; set; }
        public float[] Roll { get; set; }
        public float[] Translation { get; set; }
        public float[] Expression { get; set; }
        public float[] Scale { get; set; }
        public float[] Keypoints { get; set; }
        
        public MotionInfo()
        {
            Pitch = new float[0];
            Yaw = new float[0];
            Roll = new float[0];
            Translation = new float[0];
            Expression = new float[0];
            Scale = new float[0];
            Keypoints = new float[0];
        }
    }

    /// <summary>
    /// Data structure for appearance information from AppearanceFeatureExtractorModel
    /// </summary>
    public class AppearanceInfo
    {
        public float[] Features { get; set; }
        
        public AppearanceInfo()
        {
            Features = new float[0];
        }
    }
}
