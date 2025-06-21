using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MuseTalk.Core
{
    using API;
    using Utils;

    /// <summary>
    /// Face detection result matching Python face_analysis output
    /// </summary>
    public class FaceDetectionResult
    {
        public Rect BoundingBox { get; set; }
        public Vector2[] Keypoints5 { get; set; }  // 5 keypoints from detection
        public Vector2[] Landmarks106 { get; set; }  // 106 landmarks
        public float DetectionScore { get; set; }
        
        public override string ToString()
        {
            return $"Face(bbox={BoundingBox}, conf={DetectionScore:F3})";
        }
    }

    /// <summary>
    /// Detection candidate struct for efficient processing (value type for better cache performance)
    /// </summary>
    internal struct DetectionCandidate
    {
        public float x1, y1, x2, y2;
        public float score;
        public unsafe fixed float keypoints[10]; // 5 keypoints * 2 coords
        
        public DetectionCandidate(float x1, float y1, float x2, float y2, float score)
        {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
            // keypoints will be initialized separately
        }
    }

    /// <summary>
    /// BiSeNet face parsing classes (19 total)
    /// </summary>
    public enum FaceParsingClass
    {
        Background = 0,
        Skin = 1,           // Face region
        LeftBrow = 2,
        RightBrow = 3,
        LeftEye = 4,
        RightEye = 5,
        EyeGlass = 6,
        LeftEar = 7,
        RightEar = 8,
        Earring = 9,
        Nose = 10,
        Mouth = 11,         // Lips region
        UpperLip = 12,
        LowerLip = 13,
        Neck = 14,
        Necklace = 15,
        Cloth = 16,
        Hair = 17,
        Hat = 18
    }

    /// <summary>
    /// Consolidated face analysis class that handles detection, landmark extraction, and face parsing
    /// Uses the cleaner implementation from LivePortraitInference with added face parsing capabilities
    /// </summary>
    public class FaceAnalysis : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        // ONNX Models
        private readonly InferenceSession _detFace;      // Face detection model
        private readonly InferenceSession _landmark2d106; // 106 landmark detection model
        private readonly InferenceSession _landmarkRunner; // Landmark refinement model
        private readonly InferenceSession _faceParsingSession; // Face parsing/segmentation model
        
        // Configuration
        private readonly MuseTalkConfig _config;
        private bool _disposed = false;
        
        // Detection parameters - matching Python implementation exactly
        private const float DetectionThreshold = 0.5f;
        private const float NmsThreshold = 0.4f;
        private const int InputSize = 512;
        private const int FeatMapCount = 3;
        private static readonly int[] FeatStrideFpn = { 8, 16, 32 };
        
        // Landmark parameters
        private const int LandmarkInputSize = 192;
        private const int LandmarkRunnerSize = 224;
        
        // Cache for anchor centers
        private readonly Dictionary<string, float[,]> _centerCache = new();
        
        public bool IsInitialized { get; private set; }
        
        public FaceAnalysis(MuseTalkConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            
            try
            {
                // Load all required models
                _detFace = ModelUtils.LoadModel(_config, "det_10g");
                _landmark2d106 = ModelUtils.LoadModel(_config, "2d106det");
                _landmarkRunner = ModelUtils.LoadModel(_config, "landmark");
                
                // Load face parsing model (required for complete face analysis)
                _faceParsingSession = ModelUtils.LoadModel(_config, "face_parsing");
                
                // Verify all models initialized (including face parsing)
                bool allInitialized = _detFace != null && _landmark2d106 != null && _landmarkRunner != null && _faceParsingSession != null;
                
                if (!allInitialized)
                {
                    var failedModels = new List<string>();
                    if (_detFace == null) failedModels.Add("DetFace");
                    if (_landmark2d106 == null) failedModels.Add("Landmark106");
                    if (_landmarkRunner == null) failedModels.Add("LandmarkRunner");
                    if (_faceParsingSession == null) failedModels.Add("FaceParsing");
                    
                    throw new InvalidOperationException($"Failed to initialize face analysis models: {string.Join(", ", failedModels)}");
                }
                
                IsInitialized = true;
                Logger.Log("[FaceAnalysis] Initialized successfully");
            }
            catch (Exception e)
            {
                Logger.LogError($"[FaceAnalysis] Failed to initialize: {e.Message}");
                IsInitialized = false;
                throw;
            }
        }
        
        /// <summary>
        /// Analyze faces in image - main entry point
        /// Returns list of detected faces with landmarks, sorted by area (largest first)
        /// </summary>
        /// <summary>
        /// EXACT MATCH to LivePortraitInference.FaceAnalysis method
        /// </summary>
        public List<FaceDetectionResult> AnalyzeFaces(Frame frame, int maxFaces = -1)
        {
            if (!IsInitialized)
                throw new InvalidOperationException("FaceAnalysis not initialized");
                
            if (frame.data == null || frame.width <= 0 || frame.height <= 0)
                throw new ArgumentException("Invalid image parameters");
            
            // Process detection results exactly as in LivePortraitInference
            var faces = DetectFaces(frame);
            
            // Get landmarks for each face - EXACT MATCH to original
            var finalFaces = new List<FaceDetectionResult>();
            foreach (var face in faces)
            {
                var landmarks = GetLandmarks(frame, face);
                face.Landmarks106 = landmarks;
                finalFaces.Add(face);
            }
            
            // Python: src_face = sorted(ret, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]), reverse=True)
            finalFaces.Sort((a, b) => 
            {
                float areaA = a.BoundingBox.width * a.BoundingBox.height;
                float areaB = b.BoundingBox.width * b.BoundingBox.height;
                return areaB.CompareTo(areaA); // Descending order
            });
            
            return finalFaces;
        }
        
        /// <summary>
        /// Detect faces in image using SCRFD model - EXACT MATCH to LivePortraitInference
        /// </summary>
        private List<FaceDetectionResult> DetectFaces(Frame frame)
        {
            // Python: face_analysis(img) - EXACT MATCH to LivePortraitInference implementation
            int pythonHeight = frame.height;  // This matches Python's img.shape[0]
            int pythonWidth = frame.width;    // This matches Python's img.shape[1]
            
            // Python: im_ratio = float(img.shape[0]) / img.shape[1]
            float imRatio = (float)pythonHeight / pythonWidth;
            
            int newHeight, newWidth;
            // Python: if im_ratio > 1: new_height = input_size; new_width = int(new_height / im_ratio)
            if (imRatio > 1)
            {
                newHeight = InputSize;
                newWidth = Mathf.FloorToInt(newHeight / imRatio);
            }
            else
            {
                // Python: else: new_width = input_size; new_height = int(new_width * im_ratio)
                newWidth = InputSize;
                newHeight = Mathf.FloorToInt(newWidth * imRatio);
            }
            
            // Python: det_scale = float(new_height) / img.shape[0]
            float detScale = (float)newHeight / pythonHeight;
            
            // Python: resized_img = cv2.resize(img, (new_width, new_height))
            var resizedImg = FrameUtils.ResizeFrame(frame, newWidth, newHeight, SamplingMode.Bilinear);
            
            // Python: det_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
            // Python: det_img[:new_height, :new_width, :] = resized_img
            var detImg = new Frame(new byte[InputSize * InputSize * 3], InputSize, InputSize);
            
            // OPTIMIZED: Fill with zeros using Array.Clear (faster than manual loop)
            Array.Clear(detImg.data, 0, detImg.data.Length);
            
            // OPTIMIZED: Copy resized image to top-left with unsafe pointers and parallelization
            // EXACT MATCH to LivePortraitInference implementation
            unsafe
            {
                fixed (byte* srcPtrFixed = resizedImg.data)
                fixed (byte* dstPtrFixed = detImg.data)
                {
                    // Capture pointers in local variables to avoid lambda closure issues
                    byte* srcPtrLocal = srcPtrFixed;
                    byte* dstPtrLocal = dstPtrFixed;
                    
                    // MAXIMUM PERFORMANCE: Parallel row copying with bulk memory operations
                    // Each row can be processed independently for perfect parallelization
                    System.Threading.Tasks.Parallel.For(0, newHeight, y =>
                    {
                        // Calculate source and destination row pointers
                        byte* srcRowPtr = srcPtrLocal + y * newWidth * 3;        // Source row (RGB24)
                        byte* dstRowPtr = dstPtrLocal + y * InputSize * 3;       // Destination row (RGB24)
                        
                        // Bulk copy entire row in one operation (much faster than pixel-by-pixel)
                        int rowBytes = newWidth * 3; // RGB24 = 3 bytes per pixel
                        Buffer.MemoryCopy(srcRowPtr, dstRowPtr, rowBytes, rowBytes);
                    });
                }
            }
            
            // Python: det_img = (det_img - 127.5) / 128
            var inputTensor = FrameUtils.FrameToTensor(detImg, 0.0078125f, -0.99609375f);
            
            // Python: output = det_face.run(None, {"input.1": det_img})
            // Use the actual input name from the model metadata - EXACT MATCH
            var inputs = new List<Tensor<float>> { inputTensor };
            var results = ModelUtils.RunModel("det_face", _detFace, inputs);
            var outputs = results.ToArray();
            
            // Process detection results exactly as in LivePortraitInference
            var faces = ProcessDetectionResults(outputs, detScale);
            
            return faces;
        }
        
        /// <summary>
        /// Python: get_landmark(img, face) - EXACT MATCH to LivePortraitInference
        /// </summary>
        private Vector2[] GetLandmarks(Frame frame, FaceDetectionResult face)
        {
            // Python: input_size = 192
            const int inputSize = 192;
            
            // Python: bbox = face["bbox"]
            var bbox = face.BoundingBox;
            
            // Bbox is already in OpenCV coordinates (top-left origin), use directly
            // Python: w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            float w = bbox.width;
            float h = bbox.height;
            
            // Python: center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            Vector2 center = new(bbox.x + w * 0.5f, bbox.y + h * 0.5f);
            
            // Python: rotate = 0
            float rotate = 0f;
            
            // Python: _scale = input_size / (max(w, h) * 1.5)
            float scale = inputSize / (Mathf.Max(w, h) * 1.5f);
            
            // Python: aimg, M = face_align(img, center, input_size, _scale, rotate)
            var (alignedFrame, transformMatrix) = FaceAlign(frame, center, inputSize, scale, rotate);
            
            // Python: aimg = aimg.transpose(2, 0, 1)  # HWC -> CHW
            // Python: aimg = np.expand_dims(aimg, axis=0)
            // Python: aimg = aimg.astype(np.float32)
            var inputTensor = FrameUtils.FrameToTensor(alignedFrame, 1.0f, 0.0f);
            
            // Python: output = landmark.run(None, {"data": aimg})
            var inputs = new List<Tensor<float>> { inputTensor };
            var results = ModelUtils.RunModel("landmark2d106", _landmark2d106, inputs);
            var output = results.First().AsTensor<float>().ToArray();
            
            // Python: pred = output[0][0]
            // Python: pred = pred.reshape((-1, 2))
            var landmarks = new Vector2[output.Length / 2];
            
            // Python: pred[:, 0:2] += 1
            // Python: pred[:, 0:2] *= input_size[0] // 2
            for (int i = 0; i < landmarks.Length; i++)
            {
                float x = output[i * 2] + 1f;
                float y = output[i * 2 + 1] + 1f;
                x *= inputSize / 2f;
                y *= inputSize / 2f;
                landmarks[i] = new Vector2(x, y);
            }
            
            // Python: IM = cv2.invertAffineTransform(M)
            // Python: pred = trans_points2d(pred, IM)
            var IM = transformMatrix.inverse;
            landmarks = MathUtils.TransPoints2D(landmarks, IM);
            
            return landmarks;
        }

        /// <summary>
        /// </summary>
        public Vector2[] LandmarkRunner(Frame img, Vector2[] lmk)
        {
            // Python: crop_dct = crop_image(img, lmk, dsize=224, scale=1.5, vy_ratio=-0.1)
            var cropSize = 224;
            var cropDct = CropImage(img, lmk, cropSize, 1.5f, -0.1f);
            var imgCrop = cropDct.ImageCrop;
            
            // Python: img_crop = img_crop / 255
            // Python: img_crop = img_crop.transpose(2, 0, 1)  # HWC -> CHW
            // Python: img_crop = np.expand_dims(img_crop, axis=0)
            // Python: img_crop = img_crop.astype(np.float32)
            var inputTensor = FrameUtils.FrameToTensor(imgCrop, 1.0f / 255.0f, 0.0f);
            
            // Python: net = models["landmark_runner"]
            // Python: output = net.run(None, {"input": img_crop})
            var inputs = new List<Tensor<float>>
            {
                inputTensor
            };
            
            var results = ModelUtils.RunModel("landmark_runner", _landmarkRunner, inputs);
            var outputs = results.ToArray();
            
            // Python: out_pts = output[2]
            var outPts = outputs[2].AsTensor<float>().ToArray();
            
            // Python: lmk = out_pts[0].reshape(-1, 2) * 224  # scale to 0-224
            var refinedLmk = new Vector2[outPts.Length / 2];
            for (int i = 0; i < refinedLmk.Length; i++)
            {
                refinedLmk[i] = new Vector2(outPts[i * 2] * cropSize, outPts[i * 2 + 1] * cropSize);
            }
            
            // Python: M = crop_dct["M_c2o"]
            // Python: lmk = lmk @ M[:2, :2].T + M[:2, 2]
            refinedLmk = TransformLandmarksWithMatrix(refinedLmk, cropDct.Transform);
            
            // UnityEngine.Object.DestroyImmediate(imgCrop);
            
            return refinedLmk;
        }
        
        /// <summary>
        /// Process detection results exactly as in LivePortraitInference
        /// </summary>
        private unsafe List<FaceDetectionResult> ProcessDetectionResults(NamedOnnxValue[] outputs, float detScale)
        {
            var validDetections = new List<DetectionCandidate>();
            
            // Process each stride level
            for (int idx = 0; idx < FeatMapCount; idx++)
            {
                var scores = outputs[idx].AsTensor<float>().ToArray();
                var bboxPreds = outputs[idx + FeatMapCount].AsTensor<float>().ToArray();
                var kpsPreds = outputs[idx + FeatMapCount * 2].AsTensor<float>().ToArray();
                
                int stride = FeatStrideFpn[idx];
                int height = InputSize / stride;
                int width = InputSize / stride;
                
                // Scale predictions by stride
                for (int i = 0; i < bboxPreds.Length; i++) bboxPreds[i] *= stride;
                for (int i = 0; i < kpsPreds.Length; i++) kpsPreds[i] *= stride;
                
                // Get anchor centers
                var anchorCenters = GetAnchorCenters(height, width, stride);
                
                // Find valid detections
                for (int i = 0; i < scores.Length; i++)
                {
                    if (scores[i] >= DetectionThreshold)
                    {
                        float centerX = anchorCenters[i, 0];
                        float centerY = anchorCenters[i, 1];
                        
                        float x1 = centerX - bboxPreds[i * 4 + 0];
                        float y1 = centerY - bboxPreds[i * 4 + 1];
                        float x2 = centerX + bboxPreds[i * 4 + 2];
                        float y2 = centerY + bboxPreds[i * 4 + 3];
                        
                        var candidate = new DetectionCandidate(x1, y1, x2, y2, scores[i]);
                        
                        // Copy keypoints (keypoints is already a fixed array in the struct)
                        unsafe
                        {
                            for (int k = 0; k < 10; k += 2)
                            {
                                candidate.keypoints[k] = centerX + kpsPreds[i * 10 + k];
                                candidate.keypoints[k + 1] = centerY + kpsPreds[i * 10 + k + 1];
                            }
                        }
                        
                        validDetections.Add(candidate);
                    }
                }
            }
            
            if (validDetections.Count == 0)
                return new List<FaceDetectionResult>();
            
            // Scale by detection scale
            float invDetScale = 1.0f / detScale;
            for (int i = 0; i < validDetections.Count; i++)
            {
                var candidate = validDetections[i];
                candidate.x1 *= invDetScale;
                candidate.y1 *= invDetScale;
                candidate.x2 *= invDetScale;
                candidate.y2 *= invDetScale;
                
                unsafe
                {
                    // keypoints is already a fixed array in the struct
                    for (int k = 0; k < 10; k++)
                    {
                        candidate.keypoints[k] *= invDetScale;
                    }
                }
                validDetections[i] = candidate;
            }
            
            // Sort by score and apply NMS
            validDetections.Sort((a, b) => b.score.CompareTo(a.score));
            var keepMask = new bool[validDetections.Count];
            ApplyNMS(validDetections, keepMask);
            
            // Convert to final results
            var faces = new List<FaceDetectionResult>();
            for (int i = 0; i < validDetections.Count; i++)
            {
                if (keepMask[i])
                {
                    var candidate = validDetections[i];
                    var face = new FaceDetectionResult
                    {
                        BoundingBox = new Rect(candidate.x1, candidate.y1, 
                            candidate.x2 - candidate.x1, candidate.y2 - candidate.y1),
                        DetectionScore = candidate.score,
                        Keypoints5 = new Vector2[5]
                    };
                    
                    unsafe
                    {
                        // keypoints is already a fixed array in the struct
                        for (int k = 0; k < 5; k++)
                        {
                            face.Keypoints5[k] = new Vector2(candidate.keypoints[k * 2], candidate.keypoints[k * 2 + 1]);
                        }
                    }
                    
                    faces.Add(face);
                }
            }
            
            return faces;
        }
        
        private float[,] GetAnchorCenters(int height, int width, int stride)
        {
            string key = $"{height}_{width}_{stride}";
            
            if (_centerCache.TryGetValue(key, out var cached))
                return cached;
            
            const int numAnchors = 2;
            int totalAnchors = height * width * numAnchors;
            var centers = new float[totalAnchors, 2];
            
            int idx = 0;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float x = w * stride;
                    float y = h * stride;
                    
                    for (int a = 0; a < numAnchors; a++)
                    {
                        centers[idx, 0] = x;
                        centers[idx, 1] = y;
                        idx++;
                    }
                }
            }
            
            if (_centerCache.Count < 100)
                _centerCache[key] = centers;
                
            return centers;
        }
        
        private void ApplyNMS(List<DetectionCandidate> candidates, bool[] keepMask)
        {
            for (int i = 0; i < candidates.Count; i++)
                keepMask[i] = true;
            
            for (int i = 0; i < candidates.Count; i++)
            {
                if (!keepMask[i]) continue;
                
                var candidateA = candidates[i];
                float areaA = (candidateA.x2 - candidateA.x1) * (candidateA.y2 - candidateA.y1);
                
                for (int j = i + 1; j < candidates.Count; j++)
                {
                    if (!keepMask[j]) continue;
                    
                    var candidateB = candidates[j];
                    
                    float intersectionX1 = Mathf.Max(candidateA.x1, candidateB.x1);
                    float intersectionY1 = Mathf.Max(candidateA.y1, candidateB.y1);
                    float intersectionX2 = Mathf.Min(candidateA.x2, candidateB.x2);
                    float intersectionY2 = Mathf.Min(candidateA.y2, candidateB.y2);
                    
                    if (intersectionX1 < intersectionX2 && intersectionY1 < intersectionY2)
                    {
                        float intersectionArea = (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1);
                        float areaB = (candidateB.x2 - candidateB.x1) * (candidateB.y2 - candidateB.y1);
                        float unionArea = areaA + areaB - intersectionArea;
                        
                        if (intersectionArea / unionArea >= NmsThreshold)
                        {
                            keepMask[j] = false;
                        }
                    }
                }
            }
        }
        
        private (Frame, Matrix4x4) FaceAlign(Frame frame, Vector2 center, int outputSize, float scale, float rotate)
        {
            float scaleRatio = scale;
            float rot = rotate * Mathf.Deg2Rad;
            
            float cosRot = Mathf.Cos(rot);
            float sinRot = Mathf.Sin(rot);
            float outputSizeHalf = outputSize * 0.5f;
            
            float m00 = scaleRatio * cosRot;
            float m01 = -scaleRatio * sinRot;
            float m02 = outputSizeHalf - center.x * m00 - center.y * m01;
            
            float m10 = scaleRatio * sinRot;
            float m11 = scaleRatio * cosRot;
            float m12 = outputSizeHalf - center.x * m10 - center.y * m11;

            float[,] M = new float[,] {
                { m00, m01, m02 },
                { m10, m11, m12 }
            };
            
            var cropped = FrameUtils.AffineTransformFrame(frame, M, outputSize, outputSize);

            var transform = new Matrix4x4
            {
                m00 = M[0, 0], m01 = M[0, 1], m02 = 0f, m03 = M[0, 2],
                m10 = M[1, 0], m11 = M[1, 1], m12 = 0f, m13 = M[1, 2],
                m20 = 0f, m21 = 0f, m22 = 1f, m23 = 0f,
                m30 = 0f, m31 = 0f, m32 = 0f, m33 = 1f
            };

            return (cropped, transform);
        }
        

        
        private CropInfo CropImage(Frame frame, Vector2[] lmk, int dsize, float scale, float vyRatio)
        {
            var (MInv, _) = EstimateSimilarTransformFromPts(lmk, dsize, scale, 0f, vyRatio, true);
            
            var imgCrop = FrameUtils.AffineTransformFrame(frame, MInv, dsize, dsize);
            var ptCrop = MathUtils.TransformPts(lmk, MInv);
            
            var Mo2c = MathUtils.GetCropTransform(MInv);
            var Mc2o = Mo2c.inverse;
            
            return new CropInfo
            {
                ImageCrop = imgCrop,
                Transform = Mc2o,
                InverseTransform = Mo2c,
                LandmarksCrop = ptCrop
            };
        }
        
        private (float[,], float[,]) EstimateSimilarTransformFromPts(Vector2[] pts, int dsize, float scale, float vxRatio, float vyRatio, bool flagDoRot)
        {
            var (center, size, angle) = ParseRectFromLandmark(pts, scale, true, vxRatio, vyRatio, false);
            
            float s = dsize / size.x;
            Vector2 tgtCenter = new(dsize / 2f, dsize / 2f);
            
            float[,] MInv;
            
            if (flagDoRot)
            {
                float costheta = Mathf.Cos(angle);
                float sintheta = Mathf.Sin(angle);
                float cx = center.x, cy = center.y;
                float tcx = tgtCenter.x, tcy = tgtCenter.y;
                
                MInv = new float[,] {
                    { s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy) },
                    { -s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy) }
                };
            }
            else
            {
                MInv = new float[,] {
                    { s, 0, tgtCenter.x - s * center.x },
                    { 0, s, tgtCenter.y - s * center.y }
                };
            }
            
            var MInvH = new float[3, 3] {
                { MInv[0, 0], MInv[0, 1], MInv[0, 2] },
                { MInv[1, 0], MInv[1, 1], MInv[1, 2] },
                { 0f, 0f, 1f }
            };
            
            var M = MathUtils.InvertMatrix3x3(MInvH);
            var M2x3 = new float[,] {
                { M[0, 0], M[0, 1], M[0, 2] },
                { M[1, 0], M[1, 1], M[1, 2] }
            };

            return (MInv, M2x3);
        }
        
        private (Vector2, Vector2, float) ParseRectFromLandmark(Vector2[] pts, float scale, bool needSquare, float vxRatio, float vyRatio, bool useDegFlag)
        {
            var pt2 = ParsePt2FromPtX(pts, true);
            
            Vector2 uy = pt2[1] - pt2[0];
            float l = uy.magnitude;
            
            if (l <= 1e-3f)
            {
                uy = new Vector2(0f, 1f);
            }
            else
            {
                uy /= l;
            }
            
            Vector2 ux = new(uy.y, -uy.x);
            
            float angle = Mathf.Acos(ux.x);
            if (ux.y < 0)
            {
                angle = -angle;
            }
            
            float[,] M = new float[,] { { ux.x, ux.y }, { uy.x, uy.y } };
            
            Vector2 center0 = Vector2.zero;
            for (int i = 0; i < pts.Length; i++)
            {
                center0 += pts[i];
            }
            center0 /= pts.Length;
            
            Vector2[] rpts = new Vector2[pts.Length];
            for (int i = 0; i < pts.Length; i++)
            {
                Vector2 centered = pts[i] - center0;
                rpts[i] = new Vector2(
                    centered.x * M[0, 0] + centered.y * M[1, 0],
                    centered.x * M[0, 1] + centered.y * M[1, 1]
                );
            }
            
            Vector2 ltPt = new(float.MaxValue, float.MaxValue);
            Vector2 rbPt = new(float.MinValue, float.MinValue);
            
            for (int i = 0; i < rpts.Length; i++)
            {
                if (rpts[i].x < ltPt.x) ltPt.x = rpts[i].x;
                if (rpts[i].y < ltPt.y) ltPt.y = rpts[i].y;
                if (rpts[i].x > rbPt.x) rbPt.x = rpts[i].x;
                if (rpts[i].y > rbPt.y) rbPt.y = rpts[i].y;
            }
            
            Vector2 center1 = (ltPt + rbPt) / 2f;
            Vector2 size = rbPt - ltPt;
            
            if (needSquare)
            {
                float m = Mathf.Max(size.x, size.y);
                size.x = m;
                size.y = m;
            }
            
            size *= scale;
            
            Vector2 center = center0 + ux * center1.x + uy * center1.y;
            center = center + ux * (vxRatio * size.x) + uy * (vyRatio * size.y);
            
            if (useDegFlag)
            {
                angle *= Mathf.Rad2Deg;
            }
            
            return (center, size, angle);
        }
        
        private Vector2[] ParsePt2FromPtX(Vector2[] pts, bool useLip)
        {
            var pt2 = ParsePt2FromPt106(pts, useLip);
            
            if (!useLip)
            {
                Vector2 v = pt2[1] - pt2[0];
                pt2[1] = new Vector2(pt2[0].x - v.y, pt2[0].y + v.x);
            }
            
            return pt2;
        }
        
        private Vector2[] ParsePt2FromPt106(Vector2[] pt106, bool useLip)
        {
            Vector2 ptLeftEye = (pt106[33] + pt106[35] + pt106[40] + pt106[39]) / 4f;
            Vector2 ptRightEye = (pt106[87] + pt106[89] + pt106[94] + pt106[93]) / 4f;
            
            Vector2[] pt2;
            
            if (useLip)
            {
                Vector2 ptCenterEye = (ptLeftEye + ptRightEye) / 2f;
                Vector2 ptCenterLip = (pt106[52] + pt106[61]) / 2f;
                pt2 = new Vector2[] { ptCenterEye, ptCenterLip };
            }
            else
            {
                pt2 = new Vector2[] { ptLeftEye, ptRightEye };
            }
            
            return pt2;
        }
        
        private Vector2[] TransformLandmarksWithMatrix(Vector2[] landmarks, Matrix4x4 transform)
        {
            var result = new Vector2[landmarks.Length];
            for (int i = 0; i < landmarks.Length; i++)
            {
                var transformed = transform.MultiplyPoint3x4(new Vector3(landmarks[i].x, landmarks[i].y, 0));
                result[i] = new Vector2(transformed.x, transformed.y);
            }
            return result;
        }
        
        /// <summary>
        /// Get landmark and bbox using hybrid SCRFD+106landmark approach (matches InsightFaceHelper logic exactly)
        /// Compatible with MuseTalkInference API
        /// </summary>
        public (List<Vector4>, List<Frame>) GetLandmarkAndBbox(List<Frame> frames, int bboxShift = 0)
        {
            var coordsList = new List<Vector4>();
            var framesList = new List<Frame>();
            var CoordPlaceholder = Vector4.zero; // Matching InsightFaceHelper.CoordPlaceholder
            
            if (!IsInitialized)
            {
                Logger.LogError("[FaceAnalysis] Models not initialized");
                // Return placeholder coordinates for all frames
                for (int i = 0; i < frames.Count; i++)
                {
                    coordsList.Add(CoordPlaceholder);
                }
                return (coordsList, framesList);
            }
            
            Logger.Log($"[FaceAnalysis] Processing {frames.Count} images with hybrid SCRFD+106landmark approach");
            
            var averageRangeMinus = new List<float>();
            var averageRangePlus = new List<float>();
            
            for (int idx = 0; idx < frames.Count; idx++)
            {
                var frame = frames[idx];
                
                // Step 1: Detect faces using SCRFD (matching InsightFaceHelper exactly)
                var faces = DetectFaces(frame);   

                if (faces.Count == 0)
                {
                    Logger.LogWarning($"[FaceAnalysis] No face detected in image {idx} ({frame.width}x{frame.height})");
                    coordsList.Add(CoordPlaceholder);
                    continue;
                }
                
                // Get the best detection
                var face = faces[0];
                var bbox = face.BoundingBox;
                var scrfdKps = face.Keypoints5;
                
                // Step 2: Extract 106 landmarks using face-aligned crop (matching InsightFaceHelper flow)
                Vector2[] landmarks106;
                Vector4 finalBbox;
                
                if (scrfdKps != null && scrfdKps.Length >= 5)
                {
                    // Use existing GetLandmarks method which already gives us 106 landmarks
                    landmarks106 = GetLandmarks(frame, face);
                    if (landmarks106 != null && landmarks106.Length >= 106)
                    {
                        // Calculate final bbox using hybrid approach (adapted for 106 landmarks)
                        finalBbox = CalculateHybridBbox106(landmarks106, bbox, bboxShift);
                        
                        // Calculate range information (adapted for 106 landmarks)
                        var (rangeMinus, rangePlus) = CalculateLandmarkRanges106(landmarks106);
                        averageRangeMinus.Add(rangeMinus);
                        averageRangePlus.Add(rangePlus);
                    }
                    else
                    {
                        Logger.LogWarning($"[FaceAnalysis] Failed to extract 106 landmarks for image {idx}");
                        finalBbox = CreateFallbackBbox(bbox, scrfdKps);
                    }
                }
                else
                {
                    Logger.LogWarning($"[FaceAnalysis] No SCRFD keypoints for image {idx}, using detection bbox");
                    finalBbox = CreateFallbackBbox(bbox, null);
                }

                float width_check = finalBbox.z - finalBbox.x;
                float height_check = finalBbox.w - finalBbox.y;
                if (height_check <= 0 || width_check <= 0 || finalBbox.x < 0)
                {
                    Logger.LogWarning($"[FaceAnalysis] Invalid landmark bbox: [{finalBbox.x:F1}, {finalBbox.y:F1}, {finalBbox.z:F1}, {finalBbox.w:F1}], using SCRFD bbox");
                    coordsList.Add(CreateFallbackBbox(bbox, scrfdKps));
                }
                else
                {
                    coordsList.Add(finalBbox);
                }
                
                // Store the processed data
                framesList.Add(frame);
            }
            
            return (coordsList, framesList);
        }
        
        /// <summary>
        /// Calculate hybrid bbox using landmark center + SCRFD-like dimensions (adapted for 106 landmarks)
        /// Matches InsightFaceHelper.CalculateHybridBbox exactly but using 106 landmark indices
        /// </summary>
        private Vector4 CalculateHybridBbox106(Vector2[] landmarks106, Rect originalBbox, int bboxShift)
        {
            // MATCH PYTHON EXACTLY: Get landmark center and bounds
            // landmark_center_x = np.mean(face_land_mark[:, 0])
            // landmark_center_y = np.mean(face_land_mark[:, 1])
            Vector2 landmarkCenter = Vector2.zero;
            for (int i = 0; i < landmarks106.Length; i++)
            {
                landmarkCenter += landmarks106[i];
            }
            landmarkCenter /= landmarks106.Length;
            
            // MATCH PYTHON EXACTLY: Use SCRFD detection size as reference for proper face coverage
            // fx1, fy1, fx2, fy2 = original_bbox
            // scrfd_w = fx2 - fx1
            // scrfd_h = fy2 - fy1
            float scrfdWidth = originalBbox.width;
            float scrfdHeight = originalBbox.height;
            
            // MATCH PYTHON EXACTLY: Create bbox centered on landmarks but with SCRFD-like dimensions
            // This ensures we have enough face area for blending while being landmark-accurate
            // face_w = int(scrfd_w * 0.9)  # Slightly smaller than SCRFD for precision
            // face_h = int(scrfd_h * 0.9)
            float faceWidth = scrfdWidth * 0.9f;
            float faceHeight = scrfdHeight * 0.9f;
            
            // MATCH PYTHON EXACTLY: Center the bbox on landmark center
            // lx1 = max(0, int(landmark_center_x - face_w / 2))
            // ly1 = max(0, int(landmark_center_y - face_h / 2))
            // lx2 = min(frame.shape[1], lx1 + face_w)
            // ly2 = min(frame.shape[0], ly1 + face_h)
            float x1 = Mathf.Max(0, landmarkCenter.x - faceWidth * 0.5f);
            float y1 = Mathf.Max(0, landmarkCenter.y - faceHeight * 0.5f);
            float x2 = x1 + faceWidth;
            float y2 = y1 + faceHeight;
            
            // Apply bbox shift if specified (adapted for 106 landmarks)
            if (bboxShift != 0)
            {
                // For 106 landmarks, use nose tip (around landmark 66-68 area) as reference
                // This is the 106-landmark equivalent of the 68-landmark point 29
                if (landmarks106.Length > 66)
                {
                    Vector2 noseTipCoord = landmarks106[66]; // Approximate nose tip in 106-landmark system
                    float shiftedY = noseTipCoord.y + bboxShift;
                    float yOffset = shiftedY - noseTipCoord.y;
                    y1 += yOffset;
                    y2 += yOffset;
                }
            }
            
            return new Vector4(x1, y1, x2, y2);
        }
        
        /// <summary>
        /// Calculate landmark range information (adapted for 106 landmarks)
        /// Matches InsightFaceHelper.CalculateLandmarkRanges exactly but using 106 landmark indices
        /// </summary>
        private (float rangeMinus, float rangePlus) CalculateLandmarkRanges106(Vector2[] landmarks106)
        {
            if (landmarks106.Length < 68)
                return (20f, 20f); // Default values
            
            // For 106 landmarks, use nose area landmarks for range calculation
            // Map to equivalent nose landmarks in 106-point system (approximate mapping)
            float rangeMinus = Mathf.Abs(landmarks106[67].y - landmarks106[66].y); // Nose area
            float rangePlus = Mathf.Abs(landmarks106[66].y - landmarks106[65].y);  // Nose area
            
            return (rangeMinus, rangePlus);
        }
        
        /// <summary>
        /// Create fallback bbox when landmarks fail (matches InsightFaceHelper.CreateFallbackBbox exactly)
        /// </summary>
        private Vector4 CreateFallbackBbox(Rect originalBbox, Vector2[] scrfdKps)
        {
            // fx1, fy1, fx2, fy2 = original_bbox
            // expansion_factor = 1.05
            // center_x, center_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            // scrfd_w, scrfd_h = fx2 - fx1, fy2 - fy1
            // new_w, new_h = scrfd_w * expansion_factor, scrfd_h * expansion_factor
            float expansionFactor = 1.05f;
            float centerX = originalBbox.x + originalBbox.width * 0.5f;
            float centerY = originalBbox.y + originalBbox.height * 0.5f;
            float scrfdW = originalBbox.width;
            float scrfdH = originalBbox.height;
            float newW = scrfdW * expansionFactor;
            float newH = scrfdH * expansionFactor;
            float x1 = Mathf.Max(0, centerX - newW * 0.5f);
            float y1 = Mathf.Max(0, centerY - newH * 0.5f);
            float x2 = x1 + newW;
            float y2 = y1 + newH;
            
            return new Vector4(x1, y1, x2, y2);
        }
        
        /// <summary>
        /// Crop face region with version-specific margins (matches InsightFaceHelper.CropFaceRegion exactly)
        /// Compatible with MuseTalkInference API
        /// </summary>
        public Frame CropFaceRegion(Frame frame, Vector4 bbox, string version)
        {
            if (frame.data == null)
                return new Frame(null, 0, 0);
                
            int x1 = Mathf.RoundToInt(bbox.x);
            int y1 = Mathf.RoundToInt(bbox.y);
            int x2 = Mathf.RoundToInt(bbox.z);
            int y2 = Mathf.RoundToInt(bbox.w);
            
            // Add version-specific margin (matching InsightFaceHelper exactly)
            if (version == "v15")
            {
                y2 += 10; // extra margin for v15
                y2 = Mathf.Min(y2, frame.height);
            }
            
            int cropWidth = x2 - x1;
            int cropHeight = y2 - y1;
            
            if (frame.width <= 0 || frame.height <= 0)
            {
                Logger.LogError($"[FaceAnalysis] Invalid crop dimensions: {cropWidth}x{cropHeight}");
                throw new Exception($"[FaceAnalysis] Invalid crop dimensions: {cropWidth}x{cropHeight}");
            }

            // Extract face region (matching InsightFaceHelper coordinate system)
            var croppedFrame = FrameUtils.CropFrame(frame, new Rect(x1, y1, cropWidth, cropHeight));
            
            // Resize to standard size (256x256 for MuseTalk, matching InsightFaceHelper)
            var resizedFrame = FrameUtils.ResizeFrame(croppedFrame, 256, 256, SamplingMode.Bilinear);            
            return resizedFrame;
        }
        
        /// <summary>
        /// Generate face segmentation mask using ONNX BiSeNet model from byte array
        /// OPTIMIZED: Works with byte arrays throughout the entire pipeline
        /// </summary>
        public Frame GenerateFaceSegmentationMask(Frame frame, string mode = "jaw")
        {
            if (!IsInitialized)
            {
                Logger.LogError("[FaceAnalysis] Face analysis not initialized");
                return new Frame(null, 0, 0);
            }
            
            try
            {
                // Step 1: Preprocess image for BiSeNet (512x512, normalized) directly from byte array
                var preprocessedTensor = PreprocessImageForBiSeNet(frame);
                
                // Step 2: Run ONNX inference
                var parsingResult = RunBiSeNetInference(preprocessedTensor);
                
                // Step 3: Post-process to create mask based on mode, returning byte array
                var maskFrame = PostProcessParsingResult(parsingResult, mode, frame.width, frame.height);
                
                return maskFrame;
            }
            catch (Exception e)
            {
                Logger.LogError($"[FaceAnalysis] Face parsing failed: {e.Message}");
                return new Frame(null, 0, 0);
            }
        }
        
        /// <summary>
        /// Create face mask with morphological operations returning byte array (matching Python implementation)
        /// </summary>
        public Frame CreateFaceMaskWithMorphology(Frame frame, string mode = "jaw")
        {
            var baseMaskFrame = GenerateFaceSegmentationMask(frame, mode);
            if (baseMaskFrame.data == null) return new Frame(null, 0, 0);
            
            var smoothedMaskFrame = ApplyMorphologicalOperations(baseMaskFrame, mode);
            return smoothedMaskFrame;
        }
        

        
        private DenseTensor<float> PreprocessImageForBiSeNet(Frame inputImage)
        {
            // Resize to BiSeNet input size (512x512) - now uses optimized ResizeTextureToExactSize with byte arrays
            var resizedImageData = FrameUtils.ResizeFrame(inputImage, 512, 512, SamplingMode.Bilinear);
            
            // ImageNet normalization: (pixel/255 - mean) / std for each channel
            // R: (pixel/255 - 0.485) / 0.229, G: (pixel/255 - 0.456) / 0.224, B: (pixel/255 - 0.406) / 0.225
            // Transform to: pixel * (1/255/std) + (-mean/std)
            var multipliers = new float[] 
            { 
                1.0f / (255.0f * 0.229f),  // R: 1/(255*0.229) 
                1.0f / (255.0f * 0.224f),  // G: 1/(255*0.224)
                1.0f / (255.0f * 0.225f)   // B: 1/(255*0.225)
            };
            var offsets = new float[] 
            { 
                -0.485f / 0.229f,  // R: -mean_r/std_r
                -0.456f / 0.224f,  // G: -mean_g/std_g  
                -0.406f / 0.225f   // B: -mean_b/std_b
            };
            
            return FrameUtils.FrameToTensor(resizedImageData, multipliers, offsets);
        }
        
        private unsafe int[,] RunBiSeNetInference(DenseTensor<float> inputTensor)
        {
            // Create input for ONNX model
            var inputs = new List<Tensor<float>>
            {
                inputTensor
            };
            
            // Run inference using ModelUtils for consistency
            var results = ModelUtils.RunModel("face_parsing", _faceParsingSession, inputs);
            var output = results.First().AsTensor<float>();
            
            // Convert output to segmentation map [512, 512]
            // Output shape: [1, 19, 512, 512] - 19 face parsing classes
            var parsingMap = new int[512, 512];
            
            // Get tensor data array - unfortunately ONNX tensors don't expose direct memory access
            var outputArray = output.ToArray();
            
            // Pre-calculate tensor strides for efficient pointer arithmetic
            // Tensor layout: [batch=1, classes=19, height=512, width=512]
            const int imageSize = 512 * 512;
            const int classStride = imageSize; // Elements per class channel
            
            // OPTIMIZED: Get unsafe pointer to array data outside parallel loop
            fixed (float* outputPtr = outputArray)
            {
                // Convert pointer to IntPtr to pass into parallel lambda (C# limitation workaround)
                IntPtr outputPtrAddr = new(outputPtr);
                
                // OPTIMIZED: Maximum parallelism across all 512×512 pixels (262,144-way parallelism)
                // Apply argmax operation with direct unsafe memory access and stride-based calculation
                System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
                {
                    // Convert IntPtr back to unsafe pointer inside lambda
                    float* unsafeOutputPtr = (float*)outputPtrAddr.ToPointer();
                    
                    // Calculate x, y coordinates from linear pixel index using bit operations
                    int y = pixelIndex >> 9;  // Divide by 512 (right shift 9 bits: 2^9 = 512)
                    int x = pixelIndex & 511; // Modulo 512 (bitwise AND with 511: 2^9-1 = 511)
                    
                    // Find class with maximum probability using direct unsafe memory access
                    int maxClass = 0;
                    float* pixelPtr = unsafeOutputPtr + pixelIndex; // Pointer to class 0 for this pixel
                    float maxProb = *pixelPtr; // Dereference pointer - no bounds checking!
                    
                    // OPTIMIZED: Direct pointer arithmetic for argmax (19 classes total)
                    // Check classes 1-18 using pointer arithmetic - fastest possible access
                    for (int c = 1; c < 19; c++)
                    {
                        float* classPtr = pixelPtr + c * classStride; // Pointer to class c for this pixel
                        float prob = *classPtr; // Direct memory access - no bounds checking!
                        if (prob > maxProb)
                        {
                            maxProb = prob;
                            maxClass = c;
                        }
                    }
                    
                    // Store result in parsing map
                    parsingMap[y, x] = maxClass;
                });
            }
            
            return parsingMap;
        }
        
        private unsafe Frame PostProcessParsingResult(int[,] parsingMap, string mode, int targetWidth, int targetHeight)
        {
            // Create mask data directly as byte array (RGB24: 3 bytes per pixel)
            const int maskWidth = 512;
            const int maskHeight = 512;
            int totalBytes = maskWidth * maskHeight * 3;
            var maskFrame = new Frame(new byte[totalBytes], maskWidth, maskHeight);
            
            // Pre-calculate class IDs for each mode to avoid string comparison in hot path
            bool* classLookup = stackalloc bool[19]; // 19 face parsing classes
            
            string lowerMode = mode.ToLower();
            if (lowerMode == "neck")
            {
                // Include face, lips, and neck regions
                classLookup[(int)FaceParsingClass.Skin] = true;
                classLookup[(int)FaceParsingClass.Mouth] = true;
                classLookup[(int)FaceParsingClass.UpperLip] = true;
                classLookup[(int)FaceParsingClass.LowerLip] = true;
                classLookup[(int)FaceParsingClass.Neck] = true;
            }
            else if (lowerMode == "jaw")
            {
                // Include face and mouth regions (for talking head)
                classLookup[(int)FaceParsingClass.Skin] = true;
                classLookup[(int)FaceParsingClass.Mouth] = true;
                classLookup[(int)FaceParsingClass.UpperLip] = true;
                classLookup[(int)FaceParsingClass.LowerLip] = true;
            }
            else // "raw" or default
            {
                // Include face and lip regions
                classLookup[(int)FaceParsingClass.Skin] = true;
                classLookup[(int)FaceParsingClass.Mouth] = true;
                classLookup[(int)FaceParsingClass.UpperLip] = true;
                classLookup[(int)FaceParsingClass.LowerLip] = true;
            }
            
            // OPTIMIZED: Maximum parallelism across all 512×512 pixels (262,144-way parallelism)
            // Apply mode-specific processing with stride-based coordinate calculation
            const int imageSize = 512 * 512;
            
            fixed (byte* maskPtrFixed = maskFrame.data)
            {
                // Capture pointer in local variable to avoid lambda closure issues
                byte* maskPtrLocal = maskPtrFixed;
                
                System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
                {
                    // Calculate x, y coordinates from linear pixel index using bit operations
                    int y = pixelIndex >> 9;  // Divide by 512 (right shift 9 bits: 2^9 = 512)
                    int x = pixelIndex & 511; // Modulo 512 (bitwise AND with 511: 2^9-1 = 511)
                    
                    // Get class ID from parsing map
                    int classId = parsingMap[y, x];
                    
                    // Fast lookup using pre-calculated boolean array (no switch statement)
                    bool isForeground = classId < 19 && classLookup[classId];
                    
                    // Calculate target pixel pointer using stride arithmetic (no Y-flipping needed for byte arrays)
                    byte* pixelPtr = maskPtrLocal + ((y << 9) + x) * 3; // (y * 512 + x) * 3 for RGB24
                    
                    // Set mask value directly in memory (RGB24: all channels same for grayscale)
                    byte maskValue = isForeground ? (byte)255 : (byte)0;
                    pixelPtr[0] = maskValue; // R
                    pixelPtr[1] = maskValue; // G  
                    pixelPtr[2] = maskValue; // B
                });
            }
            
            // Resize to target dimensions if needed using optimized resize
            if (targetWidth != 512 || targetHeight != 512)
            {
                var resizedMaskFrame = FrameUtils.ResizeFrame(maskFrame, targetWidth, targetHeight, SamplingMode.Bilinear);
                return resizedMaskFrame;
            }
            
            return maskFrame;
        }
        
        private unsafe Frame ApplyMorphologicalOperations(Frame frame, string mode)
        {
            if (mode.ToLower() == "jaw")
            {
                // Apply morphological operations using unsafe pointers
                var dilatedFrame = FrameUtils.ApplyDilation(frame, 3);
                var erodedFrame = FrameUtils.ApplyErosion(dilatedFrame, 2);
                
                // Apply optimized Gaussian blur
                return FrameUtils.ApplySimpleGaussianBlur(erodedFrame, 5);
            }
            else
            {
                // For other modes, just apply light smoothing
                return FrameUtils.ApplySimpleGaussianBlur(frame, 3);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _detFace?.Dispose();
                _landmark2d106?.Dispose();
                _landmarkRunner?.Dispose();
                _faceParsingSession?.Dispose();
                _disposed = true;
                Logger.Log("[FaceAnalysis] Disposed");
            }
        }
    }
} 