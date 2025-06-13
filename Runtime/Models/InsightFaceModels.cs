using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Unity.Collections.LowLevel.Unsafe;

namespace MuseTalk.Models
{
    using Utils;

    /// <summary>
    /// SCRFD face detection model from InsightFace
    /// Provides accurate face detection with 5 facial keypoints
    /// </summary>
    public class ScrfdModel : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        private InferenceSession _session;
        private string _modelPath;
        private bool _disposed = false;
        
        // Model configuration
        private readonly Vector2Int _inputSize = new(640, 640);
        private readonly float _detThresh = 0.5f; // Exactly matching Python implementation
        private readonly float _nmsThresh = 0.4f;
        private readonly float _inputMean = 127.5f;
        private readonly float _invStd = 1.0f / 128.0f;
        
        // Model architecture parameters
        private int _fmc = 3; // Feature map count
        private int[] _featStrideFpn = { 8, 16, 32 };
        private int _numAnchors = 2;
        private bool _useKps = true; // Model supports keypoints
        
        // Model I/O names (dynamically retrieved from model metadata)
        private string _inputName;
        private string[] _outputNames;
        
        // Cache for anchor centers
        private Dictionary<string, float[,]> _centerCache = new();
        
        public bool IsInitialized { get; private set; }
        
        /// <summary>
        /// Initialize SCRFD model
        /// </summary>
        public ScrfdModel(string modelPath = null)
        {
            _modelPath = modelPath ?? Path.Combine(Application.streamingAssetsPath, "MuseTalk", "det_10g.onnx");
            
            try
            {
                Logger.Log($"[ScrfdModel] Attempting to load model from: {_modelPath}");
                Logger.Log($"[ScrfdModel] File exists: {File.Exists(_modelPath)}");
                InitializeModel();
                IsInitialized = true;
                Logger.Log($"[ScrfdModel] Successfully initialized from {_modelPath}");
            }
            catch (Exception e)
            {
                Logger.LogError($"[ScrfdModel] Failed to initialize: {e.Message}");
                Logger.LogError($"[ScrfdModel] Stack trace: {e.StackTrace}");
                IsInitialized = false;
            }
        }
        
        private void InitializeModel()
        {
            if (!File.Exists(_modelPath))
                throw new FileNotFoundException($"SCRFD model not found: {_modelPath}");

            var sessionOptions = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            _session = new InferenceSession(_modelPath, sessionOptions);
            
            var inputMetadata = _session.InputMetadata;
            var outputMetadata = _session.OutputMetadata;
            
            if (inputMetadata.Count == 0)
                throw new InvalidOperationException("Model has no inputs");
                
            _inputName = inputMetadata.Keys.First();
            _outputNames = outputMetadata.Keys.ToArray();
            
            var outputs = _session.OutputMetadata;
            int outputCount = outputs.Count;
            
            _useKps = outputCount == 9 || outputCount == 15; // Models with keypoints
            
            if (outputCount == 6 || outputCount == 9)
            {
                _fmc = 3;
                _featStrideFpn = new int[] { 8, 16, 32 };
                _numAnchors = 2;
            }
            else
            {
                _fmc = 5;
                _featStrideFpn = new int[] { 8, 16, 32, 64, 128 };
                _numAnchors = 1;
            }
            
            Logger.Log($"[ScrfdModel] Model initialized");
        }
        
        /// <summary>
        /// Detect faces in texture and return bounding boxes and keypoints
        /// </summary>
        public (FaceDetection[], Vector2[][]) DetectFaces(Texture2D texture, int maxFaces = 1)
        {
            if (!IsInitialized)
            {
                Logger.LogError("[ScrfdModel] Model not initialized");
                return (new FaceDetection[0], new Vector2[0][]);
            }
            
            if (texture == null)
            {
                Logger.LogError("[ScrfdModel] Input texture is null");
                return (new FaceDetection[0], new Vector2[0][]);
            }
                
            try
            {                
                // Prepare input tensor
                var (inputTensor, detScale) = PrepareInputTensor(texture);
                
                // Run inference with dynamically retrieved input name
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
                };
                
                using var results = _session.Run(inputs);
                
                // Process outputs
                var (detections, keypoints) = ProcessOutputs(results, texture.width, texture.height, maxFaces, detScale);
                
                return (detections, keypoints);
            }
            catch (Exception e)
            {
                Logger.LogError($"[ScrfdModel] Face detection failed: {e.Message}");
                Logger.LogError($"[ScrfdModel] Stack trace: {e.StackTrace}");
                return (new FaceDetection[0], new Vector2[0][]);
            }
        }
        
        private (DenseTensor<float>, float) PrepareInputTensor(Texture2D texture)
        {
            // Calculate scaling for letterbox (matching Python exactly)
            float imRatio = (float)texture.height / texture.width;
            float modelRatio = (float)_inputSize.y / _inputSize.x;
            
            int newWidth, newHeight;
            if (imRatio > modelRatio)
            {
                newHeight = _inputSize.y;
                newWidth = Mathf.RoundToInt(newHeight / imRatio);
            }
            else
            {
                newWidth = _inputSize.x;
                newHeight = Mathf.RoundToInt(newWidth * imRatio);
            }
            
            // Calculate detection scale (like Python: det_scale = float(new_height) / img.shape[0])
            float detScale = (float)newHeight / texture.height;
            var resized = TextureUtils.ResizeTextureToExactSize(texture, newWidth, newHeight, TextureUtils.SamplingMode.Point);
            
            int letterboxWidth = _inputSize.x;
            int letterboxHeight = _inputSize.y;
            int letterboxSize = letterboxWidth * letterboxHeight * 3; // RGB24: 3 bytes per pixel
            
            var resizedPixelData = resized.GetPixelData<byte>(0);
            
            unsafe
            {
                byte* letterboxPtr = (byte*)UnsafeUtility.Malloc(letterboxSize, 4, Unity.Collections.Allocator.Temp);
                
                try
                {
                    byte* resizedPtr = (byte*)resizedPixelData.GetUnsafeReadOnlyPtr();

                    UnsafeUtility.MemClear(letterboxPtr, letterboxSize);
                    
                    System.Threading.Tasks.Parallel.For(0, newHeight, y =>
                    {
                        byte* letterboxRowPtr = letterboxPtr + y * letterboxWidth * 3;
                        byte* resizedRowPtr = resizedPtr + y * newWidth * 3;
                        
                        // Use memcpy for entire row copy (fastest possible)
                        UnsafeUtility.MemCpy(letterboxRowPtr, resizedRowPtr, newWidth * 3);
                    });
                    
                    var tensorData = new float[1 * 3 * _inputSize.y * _inputSize.x];
                    
                    // Pre-calculate constants for performance
                    int imageSize = _inputSize.y * _inputSize.x;
                    int width = _inputSize.x;
                    int height = _inputSize.y;
                    
                    // Process in CHW format (channels first) with stride-based coordinate calculation
                    System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
                    {
                        // Calculate x, y coordinates from linear pixel index using optimized arithmetic
                        int y = pixelIndex / width;  // Row index
                        int x = pixelIndex % width;  // Column index
                        
                        int unityY = height - 1 - y; // Flip Y coordinate for ONNX coordinate system
                        
                        byte* pixelPtr = letterboxPtr + (unityY * width + x) * 3; // RGB24: 3 bytes per pixel
                        
                        // Process all 3 channels for this pixel with unrolled loop for maximum performance
                        // Apply normalization: (pixel * 255.0f - mean) * invStd
                        {
                            // R channel (channel 0)
                            float normalizedR = (pixelPtr[0] - _inputMean) * _invStd;
                            tensorData[y * width + x] = normalizedR; // Channel 0 offset: 0 * imageSize
                            
                            // G channel (channel 1)  
                            float normalizedG = (pixelPtr[1] - _inputMean) * _invStd;
                            tensorData[imageSize + y * width + x] = normalizedG; // Channel 1 offset: 1 * imageSize
                            
                            // B channel (channel 2)
                            float normalizedB = (pixelPtr[2] - _inputMean) * _invStd;
                            tensorData[2 * imageSize + y * width + x] = normalizedB; // Channel 2 offset: 2 * imageSize
                        }
                    });
                    
                    UnityEngine.Object.DestroyImmediate(resized);
                    
                    var tensor = new DenseTensor<float>(tensorData, new[] { 1, 3, _inputSize.y, _inputSize.x });
                    return (tensor, detScale);
                }
                finally
                {
                    // Clean up allocated memory
                    UnsafeUtility.Free(letterboxPtr, Unity.Collections.Allocator.Temp);
                }
            }
        }
        
        private (FaceDetection[], Vector2[][]) ProcessOutputs(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, int originalWidth, int originalHeight, int maxFaces, float detScale)
        {
            var scoresList = new List<float[]>();
            var bboxesList = new List<float[,]>();
            var kpssList = new List<float[,,]>();
            
            var resultArray = results.ToArray();
            
            // Process each feature map level (FIXED: match Python exactly)
            for (int idx = 0; idx < _fmc; idx++)
            {
                var scores = resultArray[idx].AsTensor<float>();
                var bboxPreds = resultArray[idx + _fmc].AsTensor<float>();
                Tensor<float> kpsPreds = null;
                
                if (_useKps)
                    kpsPreds = resultArray[idx + _fmc * 2].AsTensor<float>();
                
                int stride = _featStrideFpn[idx];
                int height = _inputSize.y / stride;
                int width = _inputSize.x / stride;
                
                // Get anchor centers (matching Python exactly)
                var anchorCenters = GetAnchorCenters(height, width, stride);
                
                // Convert tensors to arrays for processing
                var scoreArray = scores.ToArray();
                var bboxArray = bboxPreds.ToArray();
                var kpsArray = _useKps && kpsPreds != null ? kpsPreds.ToArray() : null;
                
                for (int i = 0; i < bboxArray.Length; i++)
                {
                    bboxArray[i] *= stride;
                }
                
                if (kpsArray != null)
                {
                    for (int i = 0; i < kpsArray.Length; i++)
                    {
                        kpsArray[i] *= stride;
                    }
                }
                
                // Find valid detections
                var validIndices = new List<int>();
                
                for (int i = 0; i < scoreArray.Length; i++)
                {
                    if (scoreArray[i] >= _detThresh)
                        validIndices.Add(i);
                }
                
                if (validIndices.Count > 0)
                {
                    var validScores = new float[validIndices.Count];
                    var validBboxes = new float[validIndices.Count, 4];
                    
                    for (int i = 0; i < validIndices.Count; i++)
                    {
                        int anchorIdx = validIndices[i];
                        validScores[i] = scoreArray[anchorIdx];
                        
                        // Get anchor center coordinates
                        float cx = anchorCenters[anchorIdx, 0];
                        float cy = anchorCenters[anchorIdx, 1];
                        
                        float x1 = cx - bboxArray[anchorIdx * 4 + 0];
                        float y1 = cy - bboxArray[anchorIdx * 4 + 1];
                        float x2 = cx + bboxArray[anchorIdx * 4 + 2];
                        float y2 = cy + bboxArray[anchorIdx * 4 + 3];
                        
                        validBboxes[i, 0] = x1;
                        validBboxes[i, 1] = y1;
                        validBboxes[i, 2] = x2;
                        validBboxes[i, 3] = y2;
                    }
                    
                    scoresList.Add(validScores);
                    bboxesList.Add(validBboxes);
                    
                    if (_useKps && kpsArray != null)
                    {
                        var validKps = new float[validIndices.Count, 5, 2]; // 5 keypoints, 2 coords each
                        
                        for (int i = 0; i < validIndices.Count; i++)
                        {
                            int anchorIdx = validIndices[i];
                            float cx = anchorCenters[anchorIdx, 0];
                            float cy = anchorCenters[anchorIdx, 1];
                            
                            for (int i_dist = 0; i_dist < 10; i_dist += 2) // 10 values for 5 keypoints
                            {
                                int kp = i_dist / 2; // Keypoint index 0-4
                                
                                float px = cx + kpsArray[anchorIdx * 10 + i_dist];
                                float py = cy + kpsArray[anchorIdx * 10 + i_dist + 1];
                                
                                validKps[i, kp, 0] = px;
                                validKps[i, kp, 1] = py;
                            }
                        }
                        
                        kpssList.Add(validKps);
                    }
                }
            }
            
            if (scoresList.Count == 0)
                return (new FaceDetection[0], new Vector2[0][]);
            
            // Combine all detections and apply NMS
            return CombineAndFilter(scoresList, bboxesList, kpssList, originalWidth, originalHeight, maxFaces, detScale);
        }
        
        private float[,] GetAnchorCenters(int height, int width, int stride)
        {
            string key = $"{height}_{width}_{stride}";
            
            if (_centerCache.ContainsKey(key))
                return _centerCache[key];
            
            // Step 1: Create basic grid coordinates (matching np.mgrid[:height, :width])
            int totalAnchors = height * width;
            if (_numAnchors > 1)
                totalAnchors *= _numAnchors;
                
            var centers = new float[totalAnchors, 2];
            int idx = 0;
            
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float centerX = (float)x;
                    float centerY = (float)y;
                    
                    centerX *= stride;
                    centerY *= stride;
                    
                    // Handle multiple anchors per grid point
                    if (_numAnchors > 1)
                    {
                        for (int a = 0; a < _numAnchors; a++)
                        {
                            centers[idx, 0] = centerX;
                            centers[idx, 1] = centerY;
                            idx++;
                        }
                    }
                    else
                    {
                        centers[idx, 0] = centerX;
                        centers[idx, 1] = centerY;
                        idx++;
                    }
                }
            }
            
            if (_centerCache.Count < 100)
                _centerCache[key] = centers;
                
            return centers;
        }
        
        private (FaceDetection[], Vector2[][]) CombineAndFilter(List<float[]> scoresList, List<float[,]> bboxesList, List<float[,,]> kpssList, int originalWidth, int originalHeight, int maxFaces, float detScale)
        {
            if (scoresList.Count == 0)
                return (new FaceDetection[0], new Vector2[0][]);
            
            var allScores = new List<float>();
            var allBboxes = new List<float[]>();
            var allKps = new List<Vector2[]>();
            
            for (int level = 0; level < scoresList.Count; level++)
            {
                var scores = scoresList[level];
                var bboxes = bboxesList[level];
                
                for (int i = 0; i < scores.Length; i++)
                {
                    allScores.Add(scores[i]);
                    
                    allBboxes.Add(new float[] {
                        bboxes[i, 0] / detScale,
                        bboxes[i, 1] / detScale,
                        bboxes[i, 2] / detScale,
                        bboxes[i, 3] / detScale
                    });
                    
                    if (_useKps && kpssList.Count > level)
                    {
                        var kps = kpssList[level];
                        var keypoints = new Vector2[5];
                        
                        for (int kp = 0; kp < 5; kp++)
                        {
                            keypoints[kp] = new Vector2(
                                kps[i, kp, 0] / detScale,
                                kps[i, kp, 1] / detScale
                            );
                        }
                        allKps.Add(keypoints);
                    }
                    else
                    {
                        allKps.Add(new Vector2[0]);
                    }
                }
            }
            
            if (allScores.Count == 0)
                return (new FaceDetection[0], new Vector2[0][]);
            
            var indices = Enumerable.Range(0, allScores.Count)
                .OrderByDescending(i => allScores[i])
                .ToArray();
            
            var keepIndices = ApplyNMS(allBboxes, allScores, indices);
            
            var finalDetections = new List<FaceDetection>();
            var finalKeypoints = new List<Vector2[]>();
            
            foreach (int idx in keepIndices.Take(maxFaces))
            {
                var bbox = allBboxes[idx];
                var detection = new FaceDetection
                {
                    BoundingBox = new Rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                    Confidence = allScores[idx]
                };
                
                finalDetections.Add(detection);
                finalKeypoints.Add(allKps[idx]);
            }
            
            if (maxFaces > 0 && finalDetections.Count > maxFaces)
            {
                var prioritizedIndices = PrioritizeDetectionsByAreaAndCenter(finalDetections, originalWidth, originalHeight, maxFaces);
                var prioritizedDetections = prioritizedIndices.Select(i => finalDetections[i]).ToArray();
                var prioritizedKps = prioritizedIndices.Select(i => finalKeypoints[i]).ToArray();
                return (prioritizedDetections, prioritizedKps);
            }
            
            return (finalDetections.ToArray(), finalKeypoints.ToArray());
        }
        
        /// <summary>
        /// Apply Non-Maximum Suppression matching Python implementation exactly
        /// </summary>
        private List<int> ApplyNMS(List<float[]> bboxes, List<float> scores, int[] sortedIndices)
        {
            var keep = new List<int>();
            var remaining = new List<int>(sortedIndices);
            
            while (remaining.Count > 0)
            {
                int currentIdx = remaining[0];
                keep.Add(currentIdx);
                remaining.RemoveAt(0);
                
                if (remaining.Count == 0) break;
                
                var currentBbox = bboxes[currentIdx];
                float currentArea = (currentBbox[2] - currentBbox[0] + 1) * (currentBbox[3] - currentBbox[1] + 1);
                
                var toRemove = new List<int>();
                
                foreach (int otherIdx in remaining)
                {
                    var otherBbox = bboxes[otherIdx];
                    float otherArea = (otherBbox[2] - otherBbox[0] + 1) * (otherBbox[3] - otherBbox[1] + 1);
                    
                    float xx1 = Mathf.Max(currentBbox[0], otherBbox[0]);
                    float yy1 = Mathf.Max(currentBbox[1], otherBbox[1]);
                    float xx2 = Mathf.Min(currentBbox[2], otherBbox[2]);
                    float yy2 = Mathf.Min(currentBbox[3], otherBbox[3]);
                    
                    float w = Mathf.Max(0.0f, xx2 - xx1 + 1);
                    float h = Mathf.Max(0.0f, yy2 - yy1 + 1);
                    float intersection = w * h;
                    
                    float iou = intersection / (currentArea + otherArea - intersection);
                    
                    if (iou > _nmsThresh)
                    {
                        toRemove.Add(otherIdx);
                    }
                }
                
                foreach (int idx in toRemove)
                {
                    remaining.Remove(idx);
                }
            }
            
            return keep;
        }
        
        /// <summary>
        /// Prioritize detections by area and distance from center (matching Python exactly)
        /// </summary>
        private int[] PrioritizeDetectionsByAreaAndCenter(List<FaceDetection> detections, int imageWidth, int imageHeight, int maxNum)
        {
            float centerX = imageWidth / 2.0f;
            float centerY = imageHeight / 2.0f;
            
            var priorities = new List<(int index, float priority)>();
            
            for (int i = 0; i < detections.Count; i++)
            {
                var bbox = detections[i].BoundingBox;
                float area = bbox.width * bbox.height;
                
                float bboxCenterX = bbox.x + bbox.width / 2.0f;
                float bboxCenterY = bbox.y + bbox.height / 2.0f;
                
                float offsetX = bboxCenterX - centerX;
                float offsetY = bboxCenterY - centerY;
                float offsetDistSquared = offsetX * offsetX + offsetY * offsetY;
                
                float priority = area - offsetDistSquared * 2.0f;
                priorities.Add((i, priority));
            }
            
            // Sort by priority descending and take top maxNum
            return priorities
                .OrderByDescending(p => p.priority)
                .Take(maxNum)
                .Select(p => p.index)
                .ToArray();
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                Logger.Log("[ScrfdModel] Disposed");
            }
        }
    }
    
    /// <summary>
    /// 1k3d68 landmark detection model from InsightFace
    /// Provides 68-point facial landmarks from face crops
    /// </summary>
    public class Landmark68Model : IDisposable
    {
        private static DebugLogger Logger = new DebugLogger();
        
        private InferenceSession _session;
        private string _modelPath;
        private bool _disposed = false;
        
        // Model configuration
        private readonly Vector2Int _inputSize = new Vector2Int(192, 192);
        private readonly float _inputMean = 127.5f;
        private readonly float _invStd = 1.0f / 128.0f;
        
        // Model I/O names (dynamically retrieved from model metadata)
        private string _inputName;
        private string[] _outputNames;
        
        public bool IsInitialized { get; private set; }
        public int LandmarkCount { get; private set; } = 68;
        public int LandmarkDimension { get; private set; } = 2; // 2D landmarks
        
        public Landmark68Model(string modelPath = null)
        {
            _modelPath = modelPath ?? Path.Combine(Application.streamingAssetsPath, "MuseTalk", "1k3d68.onnx");
            
            try
            {
                Logger.Log($"[Landmark68Model] Attempting to load model from: {_modelPath}");
                Logger.Log($"[Landmark68Model] File exists: {File.Exists(_modelPath)}");
                InitializeModel();
                IsInitialized = true;
                Logger.Log($"[Landmark68Model] Successfully initialized from {_modelPath}");
            }
            catch (Exception e)
            {
                Logger.LogError($"[Landmark68Model] Failed to initialize: {e.Message}");
                Logger.LogError($"[Landmark68Model] Stack trace: {e.StackTrace}");
                IsInitialized = false;
            }
        }
        
        private void InitializeModel()
        {
            if (!File.Exists(_modelPath))
                throw new FileNotFoundException($"1k3d68 model not found: {_modelPath}");

            var sessionOptions = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            _session = new InferenceSession(_modelPath, sessionOptions);
            
            var inputMetadata = _session.InputMetadata;
            var outputMetadata = _session.OutputMetadata;
            
            if (inputMetadata.Count == 0)
                throw new InvalidOperationException("Model has no inputs");
                
            _inputName = inputMetadata.Keys.First();
            _outputNames = outputMetadata.Keys.ToArray();
            
            var outputMeta = _session.OutputMetadata.First().Value;
            var outputShape = outputMeta.Dimensions;
            
            if (outputShape.Length > 1 && outputShape[1] >= 3000)
            {
                LandmarkDimension = 3; // 3D landmarks
            }
            
            Logger.Log($"[Landmark68Model] Model configuration: {LandmarkCount} landmarks, {LandmarkDimension}D");
        }
        
        /// <summary>
        /// Extract 68-point facial landmarks matching Python InsightFaceLandmark.get_landmarks EXACTLY
        /// </summary>
        public Vector2[] GetLandmarks(Texture2D texture, Rect bbox, Vector2[] scrfdKeypoints = null)
        {
            if (!IsInitialized || texture == null)
                return new Vector2[0];
                
            try
            {
                float x1 = bbox.x;
                float y1 = bbox.y; 
                float x2 = bbox.x + bbox.width;
                float y2 = bbox.y + bbox.height;
                float w = x2 - x1;
                float h = y2 - y1;
                Vector2 center = new((x1 + x2) / 2f, (y1 + y2) / 2f);
                float scale = (float)_inputSize.x / (Mathf.Max(w, h) * 1.5f);
                float rotate = 0f;
                
                var (transformedCrop, transformMatrix) = ApplyInsightFaceTransform(texture, center, _inputSize.x, scale, rotate);
                
                var rawLandmarks = ExtractLandmarks(transformedCrop);
                var processedLandmarks = ProcessInsightFaceLandmarks(rawLandmarks);
                var finalLandmarks = TransformLandmarksToOriginal(processedLandmarks, transformMatrix);
                
                UnityEngine.Object.DestroyImmediate(transformedCrop);
                return finalLandmarks;
            }
            catch (Exception e)
            {
                Logger.LogError($"[Landmark68Model] Landmark extraction failed: {e.Message}");
                return new Vector2[0];
            }
        }
        
        /// <summary>
        /// Apply transformation matching official InsightFace face_align.transform
        /// </summary>
        private (Texture2D, Matrix4x4) ApplyInsightFaceTransform(Texture2D img, Vector2 center, int outputSize, float scale, float rotation)
        {
            float scaleRatio = scale;
            float rot = rotation * Mathf.PI / 180.0f;
            
            // Calculate transformed center
            float cx = center.x * scaleRatio;
            float cy = center.y * scaleRatio;
            
            float cos_rot = Mathf.Cos(rot);
            float sin_rot = Mathf.Sin(rot);
            
            Matrix4x4 M = Matrix4x4.identity;
            M.m00 = scaleRatio * cos_rot;
            M.m01 = -scaleRatio * sin_rot;
            M.m02 = -cx * cos_rot + cy * sin_rot + outputSize / 2f;
            M.m10 = scaleRatio * sin_rot;
            M.m11 = scaleRatio * cos_rot;
            M.m12 = -cx * sin_rot - cy * cos_rot + outputSize / 2f;
            
            var transformedTexture = ApplyAffineTransform(img, M, outputSize);
            return (transformedTexture, M);
        }
        
        /// <summary>
        /// Apply affine transformation to texture (matching cv2.warpAffine)
        /// OPTIMIZED: Uses unsafe pointers, parallelization, and direct memory operations for maximum performance
        /// </summary>
        private unsafe Texture2D ApplyAffineTransform(Texture2D source, Matrix4x4 M, int outputSize)
        {
            // Create result texture with RGB24 format for efficiency
            var result = new Texture2D(outputSize, outputSize, TextureFormat.RGB24, false);
            
            // Get pixel data as byte arrays for direct memory access
            var sourcePixelData = source.GetPixelData<byte>(0);
            var resultPixelData = result.GetPixelData<byte>(0);
            
            // Get unsafe pointers for direct memory operations
            byte* sourcePtr = (byte*)sourcePixelData.GetUnsafeReadOnlyPtr();
            byte* resultPtr = (byte*)resultPixelData.GetUnsafePtr();
            
            int sourceWidth = source.width;
            int sourceHeight = source.height;
            int totalPixels = outputSize * outputSize;
            
            // Pre-calculate transformation matrix elements for performance
            float m00 = M.m00, m01 = M.m01, m02 = M.m02;
            float m10 = M.m10, m11 = M.m11, m12 = M.m12;
            
            // OPTIMIZED: Maximum parallelism across all output pixels
            System.Threading.Tasks.Parallel.For(0, totalPixels, pixelIndex =>
            {
                // Calculate x, y coordinates from linear pixel index using optimized arithmetic
                int y = pixelIndex / outputSize;  // Row index
                int x = pixelIndex % outputSize;  // Column index
                
                // Apply inverse transformation using pre-calculated matrix elements
                float srcX = m00 * x + m01 * y + m02;
                float srcY = m10 * x + m11 * y + m12;
                
                // OPTIMIZED: Direct unsafe bilinear sampling
                byte r, g, b;
                BilinearSampleUnsafe(sourcePtr, sourceWidth, sourceHeight, srcX, srcY, out r, out g, out b);
                
                // Calculate target pixel pointer using stride arithmetic
                byte* targetPixel = resultPtr + (y * outputSize + x) * 3;
                
                // Direct memory write (RGB24: 3 bytes per pixel)
                targetPixel[0] = r; // R
                targetPixel[1] = g; // G
                targetPixel[2] = b; // B
            });
            
            // Apply changes to texture (no need for SetPixels since we wrote directly to pixel data)
            result.Apply();
            return result;
        }
        
        /// <summary>
        /// Unsafe optimized bilinear sampling using direct byte pointer access
        /// OPTIMIZED: Direct pointer arithmetic for maximum performance, no Color allocations
        /// </summary>
        private static unsafe void BilinearSampleUnsafe(byte* sourcePtr, int width, int height, float x, float y, out byte r, out byte g, out byte b)
        {
            // Bounds check - return black if outside valid sampling area
            if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1)
            {
                r = g = b = 0;
                return;
            }
            
            // Get integer and fractional parts for bilinear interpolation
            int x0 = (int)x;  // Faster than Mathf.FloorToInt for positive values
            int y0 = (int)y;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float fx = x - x0;
            float fy = y - y0;
            float invFx = 1.0f - fx;
            float invFy = 1.0f - fy;
            
            // Pre-calculate bilinear interpolation weights
            float w00 = invFx * invFy; // Top-left weight
            float w10 = fx * invFy;    // Top-right weight
            float w01 = invFx * fy;    // Bottom-left weight
            float w11 = fx * fy;       // Bottom-right weight
            
            // Calculate source pixel pointers using stride arithmetic (RGB24: 3 bytes per pixel)
            byte* p00 = sourcePtr + (y0 * width + x0) * 3; // Top-left
            byte* p10 = sourcePtr + (y0 * width + x1) * 3; // Top-right
            byte* p01 = sourcePtr + (y1 * width + x0) * 3; // Bottom-left
            byte* p11 = sourcePtr + (y1 * width + x1) * 3; // Bottom-right
            
            // OPTIMIZED: Direct byte interpolation with unrolled RGB channels
            // R channel
            float rFloat = p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11;
            r = (byte)Mathf.Clamp(rFloat, 0f, 255f);
            
            // G channel
            float gFloat = p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11;
            g = (byte)Mathf.Clamp(gFloat, 0f, 255f);
            
            // B channel
            float bFloat = p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11;
            b = (byte)Mathf.Clamp(bFloat, 0f, 255f);
        }
        
        /// <summary>
        /// Process landmarks matching official InsightFace coordinate processing exactly
        /// </summary>
        private Vector2[] ProcessInsightFaceLandmarks(Vector2[] rawLandmarks)
        {
            var processed = new Vector2[rawLandmarks.Length];
            float scaleFactor = _inputSize.x / 2f; // input_size[0] // 2
            
            for (int i = 0; i < rawLandmarks.Length; i++)
            {
                Vector2 shifted = rawLandmarks[i] + Vector2.one;
                processed[i] = shifted * scaleFactor;
            }
            return processed;
        }
        
        private Rect CreateFaceAlignedBbox(Vector2[] scrfdKeypoints, Texture2D texture)
        {
            // SCRFD keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth]
            Vector2 leftEye = scrfdKeypoints[0];
            Vector2 rightEye = scrfdKeypoints[1];
            Vector2 nosetip = scrfdKeypoints[2];
            
            // Calculate face center and scale using eye distance
            Vector2 eyeCenter = (leftEye + rightEye) * 0.5f;
            float eyeDistance = Vector2.Distance(rightEye, leftEye);
            
            // Create face-aligned bbox using empirical factors
            float faceScale = eyeDistance * 2.2f; // Empirical factor for good face coverage
            
            // Center around eye center but shift down for mouth coverage
            Vector2 cropCenter = new Vector2(
                eyeCenter.x,
                eyeCenter.y + eyeDistance * 0.3f // Shift down for mouth coverage
            );
            
            // Create square bbox
            float halfSize = faceScale * 0.5f;
            Rect alignedBbox = new(
                Mathf.Max(0, cropCenter.x - halfSize),
                Mathf.Max(0, cropCenter.y - halfSize),
                Mathf.Min(texture.width, faceScale),
                Mathf.Min(texture.height, faceScale)
            );
            
            // Ensure square dimensions
            float minDim = Mathf.Min(alignedBbox.width, alignedBbox.height);
            alignedBbox.width = minDim;
            alignedBbox.height = minDim;
            
            return alignedBbox;
        }
        
        private Vector2[] ExtractLandmarks(Texture2D alignedCrop)
        {
            var inputTensor = CreateLandmarkInputTensor(alignedCrop);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };
            
            // Run inference
            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>();
            var outputArray = output.ToArray();
            
            Vector2[] rawLandmarks;
            
            if (outputArray.Length >= 3000) // 3D landmarks
            {
                int numLandmarks = outputArray.Length / 3;
                rawLandmarks = new Vector2[numLandmarks];
                
                for (int i = 0; i < numLandmarks; i++)
                {
                    float x = outputArray[i * 3];
                    float y = outputArray[i * 3 + 1];
                    rawLandmarks[i] = new Vector2(x, y);
                }
                
                LandmarkDimension = 3; // Update dimension info
            }
            else // 2D landmarks
            {
                int numLandmarks = outputArray.Length / 2;
                rawLandmarks = new Vector2[numLandmarks];
                
                for (int i = 0; i < numLandmarks; i++)
                {
                    float x = outputArray[i * 2];
                    float y = outputArray[i * 2 + 1];
                    rawLandmarks[i] = new Vector2(x, y);
                }
                LandmarkDimension = 2; // Update dimension info
            }
            
            // Take only the last 68 landmarks if needed
            if (rawLandmarks.Length > LandmarkCount)
            {
                var trimmed = new Vector2[LandmarkCount];
                Array.Copy(rawLandmarks, rawLandmarks.Length - LandmarkCount, trimmed, 0, LandmarkCount);
                rawLandmarks = trimmed;
                Logger.Log($"[Landmark68Model] Trimmed to last {LandmarkCount} landmarks");
            }
            
            Logger.Log($"[Landmark68Model] Extracted {rawLandmarks.Length} RAW landmarks from model");
            return rawLandmarks;
        }
        
        /// <summary>
        /// Create input tensor for landmark model
        /// OPTIMIZED: Uses unsafe pointers and parallelization for maximum performance
        /// </summary>
        private unsafe DenseTensor<float> CreateLandmarkInputTensor(Texture2D texture)
        {
            // Ensure texture is the right size using optimized resize
            if (texture.width != _inputSize.x || texture.height != _inputSize.y)
            {
                texture = TextureUtils.ResizeTextureToExactSize(texture, _inputSize.x, _inputSize.y, TextureUtils.SamplingMode.Point);
            }
            
            var tensorData = new float[1 * 3 * _inputSize.y * _inputSize.x];
            var pixelData = texture.GetPixelData<byte>(0);
            
            byte* pixelPtr = (byte*)pixelData.GetUnsafeReadOnlyPtr();
            
            int imageSize = _inputSize.y * _inputSize.x;
            int width = _inputSize.x;
            int height = _inputSize.y;
            
            // Process in CHW format (channels first) with stride-based coordinate calculation
            System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
            {
                // Calculate x, y coordinates from linear pixel index
                int y = pixelIndex / width;  // Row index
                int x = pixelIndex % width;  // Column index
                
                int unityY = height - 1 - y; // Flip Y coordinate for ONNX coordinate system
                
                byte* pixelBytePtr = pixelPtr + (unityY * width + x) * 3; // RGB24: 3 bytes per pixel
                
                // Process all 3 channels for this pixel with unrolled loop for maximum performance
                // Apply normalization: (pixel - mean) * invStd
                {
                    // R channel (channel 0)
                    float normalizedR = (pixelBytePtr[0] - _inputMean) * _invStd;
                    tensorData[y * width + x] = normalizedR; // Channel 0 offset: 0 * imageSize
                    
                    // G channel (channel 1)  
                    float normalizedG = (pixelBytePtr[1] - _inputMean) * _invStd;
                    tensorData[imageSize + y * width + x] = normalizedG; // Channel 1 offset: 1 * imageSize
                    
                    // B channel (channel 2)
                    float normalizedB = (pixelBytePtr[2] - _inputMean) * _invStd;
                    tensorData[2 * imageSize + y * width + x] = normalizedB; // Channel 2 offset: 2 * imageSize
                }
            });
            
            return new DenseTensor<float>(tensorData, new[] { 1, 3, _inputSize.y, _inputSize.x });
        }
        
        private Vector2[] TransformLandmarksToOriginal(Vector2[] landmarks, Matrix4x4 transformMatrix)
        {
            // CRITICAL FIX: Match Python _trans_points2d exactly
            // Python: IM = cv2.invertAffineTransform(M)
            // Python: pred = self._trans_points(pred, IM)
            
            var originalLandmarks = new Vector2[landmarks.Length];
            
            // Create 2x3 affine matrix from 4x4 (matching Python cv2.invertAffineTransform)
            float[,] M = new float[2, 3] {
                { transformMatrix.m00, transformMatrix.m01, transformMatrix.m02 },
                { transformMatrix.m10, transformMatrix.m11, transformMatrix.m12 }
            };
            
            // Compute inverse affine transformation (matching cv2.invertAffineTransform)
            float det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0];
            if (Mathf.Abs(det) < 1e-6f)
            {
                Logger.LogError("[Landmark68Model] Singular transformation matrix, cannot invert");
                return landmarks; // Return original if can't invert
            }
            
            float[,] IM = new float[2, 3];
            IM[0, 0] = M[1, 1] / det;
            IM[0, 1] = -M[0, 1] / det;
            IM[1, 0] = -M[1, 0] / det;
            IM[1, 1] = M[0, 0] / det;
            IM[0, 2] = (M[0, 1] * M[1, 2] - M[1, 1] * M[0, 2]) / det;
            IM[1, 2] = (M[1, 0] * M[0, 2] - M[0, 0] * M[1, 2]) / det;
            
            // Apply inverse transformation
            for (int i = 0; i < landmarks.Length; i++)
            {
                Vector2 pt = landmarks[i];
                
                float newX = IM[0, 0] * pt.x + IM[0, 1] * pt.y + IM[0, 2];
                float newY = IM[1, 0] * pt.x + IM[1, 1] * pt.y + IM[1, 2];
                
                originalLandmarks[i] = new Vector2(newX, newY);
            }
            
            Logger.Log($"[Landmark68Model] Transformed {landmarks.Length} landmarks back to original coordinates");
            return originalLandmarks;
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                Logger.Log("[Landmark68Model] Disposed");
            }
        }
    }
    
    /// <summary>
    /// Face detection result from SCRFD
    /// </summary>
    [Serializable]
    public class FaceDetection
    {
        public Rect BoundingBox;
        public float Confidence;
        
        public override string ToString()
        {
            return $"Face(bbox={BoundingBox}, conf={Confidence:F3})";
        }
    }
} 