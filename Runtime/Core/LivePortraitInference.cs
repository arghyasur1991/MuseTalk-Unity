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
    using System.Diagnostics;
    using Debug = UnityEngine.Debug;

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
        public Texture2D MaskTemplate { get; set; } // Python: mask_crop from mask_template.png
    }

    /// <summary>
    /// Face detection result matching Python face_analysis output
    /// </summary>
    public class FaceDetectionResult
    {
        public Rect BoundingBox { get; set; }
        public Vector2[] Keypoints5 { get; set; }  // 5 keypoints from detection
        public Vector2[] Landmarks106 { get; set; }  // 106 landmarks
        public float DetectionScore { get; set; }
    }

    /// <summary>
    /// Crop information matching Python crop_info
    /// </summary>
    public class CropInfo
    {
        public Texture2D ImageCrop { get; set; }
        public Texture2D ImageCrop256x256 { get; set; }
        public Vector2[] LandmarksCrop { get; set; }
        public Vector2[] LandmarksCrop256x256 { get; set; }
        public Matrix4x4 Transform { get; set; }
        public Matrix4x4 InverseTransform { get; set; }
    }

    /// <summary>
    /// Core LivePortrait inference engine that matches onnx_inference.py EXACTLY
    /// ALL OPERATIONS ON MAIN THREAD FOR CORRECTNESS FIRST
    /// COMPLETELY SELF-SUFFICIENT - NO EXTERNAL DEPENDENCIES
    /// </summary>
    public class LivePortraitInference : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        // LivePortrait ONNX models - matches Python models dict exactly
        private InferenceSession _detFace;  // face detection
        private InferenceSession _landmark2d106;  // 106 landmark detection
        private InferenceSession _landmarkRunner;  // landmark refinement
        private InferenceSession _appearanceFeatureExtractor;  // feature extraction
        private InferenceSession _motionExtractor;  // motion parameters
        private InferenceSession _stitching;  // keypoint stitching
        private InferenceSession _warpingSpade;  // neural warping
        
        // Face analysis (reuse existing InsightFace)
        private InsightFaceHelper _insightFaceHelper;

        private Texture2D _debugImage = null;
        
        // Configuration
        private MuseTalkConfig _config;
        private bool _initialized = false;
        private bool _disposed = false;
        
        // State management - matches Python self.pred_info
        private LivePortraitPredInfo _predInfo;
        
        // Composite flag - matches Python self.flg_composite
        private bool _flgComposite = false;
        
        // Mask template - matches Python self.mask_crop
        private Texture2D _maskTemplate;
        
        public bool IsInitialized => _initialized;
        
        public LivePortraitInference(MuseTalkConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            
            try
            {
                InitializeModels();
                
                // Initialize prediction state - matches Python self.pred_info = {'lmk':None, 'x_d_0_info':None}
                _predInfo = new LivePortraitPredInfo
                {
                    Landmarks = null,
                    InitialMotionInfo = null
                };
                
                // Load mask template - matches Python self.mask_crop = cv2.imread('mask_template.png')
                _maskTemplate = ModelUtils.LoadMaskTemplate(_config);
                
                _initialized = true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[LivePortraitInference] Failed to initialize: {e.Message}");
                _initialized = false;
            }
        }
        
        private void InitializeModels()
        {
            // Load all ONNX models exactly as in Python
            _detFace = ModelUtils.LoadModel(_config, "det_10g");
            _landmark2d106 = ModelUtils.LoadModel(_config, "2d106det");
            _landmarkRunner = ModelUtils.LoadModel(_config, "landmark");
            _appearanceFeatureExtractor = ModelUtils.LoadModel(_config, "appearance_feature_extractor");
            _motionExtractor = ModelUtils.LoadModel(_config, "motion_extractor");
            _stitching = ModelUtils.LoadModel(_config, "stitching");
            _warpingSpade = ModelUtils.LoadModel(_config, "warping_spade");
            
            // Reuse existing InsightFace helper
            _insightFaceHelper = new InsightFaceHelper(_config);
            
            // Verify all models initialized
            bool allInitialized = _appearanceFeatureExtractor != null &&
                                 _motionExtractor != null &&
                                 _warpingSpade != null &&
                                 _stitching != null &&
                                 _landmarkRunner != null &&
                                 _landmark2d106 != null &&
                                 _detFace != null &&
                                 _insightFaceHelper.IsInitialized;
            
            if (!allInitialized)
            {
                var failedModels = new List<string>();
                if (_appearanceFeatureExtractor == null) failedModels.Add("AppearanceExtractor");
                if (_motionExtractor == null) failedModels.Add("MotionExtractor");
                if (_warpingSpade == null) failedModels.Add("WarpingSPADE");
                if (_stitching == null) failedModels.Add("Stitching");
                if (_landmarkRunner == null) failedModels.Add("LandmarkRunner");
                if (_landmark2d106 == null) failedModels.Add("Landmark106");
                if (_detFace == null) failedModels.Add("DetFace");
                if (!_insightFaceHelper.IsInitialized) failedModels.Add("InsightFace");
                
                throw new InvalidOperationException($"Failed to initialize models: {string.Join(", ", failedModels)}");
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
                
            // Store mask template - matches Python self.mask_crop
            // Use input mask template if provided, otherwise use the one loaded during initialization
            if (input.MaskTemplate != null)
            {
                _maskTemplate = input.MaskTemplate;
            }
            else if (_maskTemplate != null)
            {
                // Using mask template loaded during initialization
            }
            else
            {
                Debug.LogWarning("[LivePortraitInference] No mask template available, will use default circular mask");
            }
                
            try
            {
                var start = Stopwatch.StartNew();
                // Generate frames
                var generatedFrames = new List<Texture2D>();
                var (srcImgData, srcImgWidth, srcImgHeight) = SrcPreprocess(input.SourceImage);
                var srcImgElapsed = start.ElapsedMilliseconds;
                Debug.Log($"[LivePortraitInference] SrcPreprocess took {srcImgElapsed}ms");
                
                // CRITICAL FIX: Keep reference to preprocessed source image for pasteback
                // In Python, src_img is used for pasteback - make sure it's valid
                // TODO: Convert CropSrcImage to work with byte arrays
                var srcImgTexture = BytesToTexture2D(srcImgData, srcImgWidth, srcImgHeight);
                
                // Python: crop_info = crop_src_image(self.models, src_img)
                var cropInfo = CropSrcImage(srcImgTexture);
                // Python: img_crop_256x256 = crop_info["img_crop_256x256"]
                // Python: I_s = preprocess(img_crop_256x256)
                var Is = Preprocess(cropInfo.ImageCrop256x256);
                
                // Python: x_s_info = get_kp_info(self.models, I_s)
                var xSInfo = GetKpInfo(Is);
                
                // Python: R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
                var Rs = GetRotationMatrix(xSInfo.Pitch, xSInfo.Yaw, xSInfo.Roll);
                
                // Python: f_s = extract_feature_3d(self.models, I_s)
                var fs = ExtractFeature3d(Is);
                
                // Python: x_s = transform_keypoint(x_s_info)
                var xs = TransformKeypoint(xSInfo);
                
                // Python: prepare for pasteback
                // Python: mask_ori = prepare_paste_back(self.mask_crop, crop_info["M_c2o"], dsize=(src_img.shape[1], src_img.shape[0]))
                var maskOri = PreparePasteBack(cropInfo.Transform, srcImgWidth, srcImgHeight);

                var maxFrames = 1;

                // For debugging, only generate 1 frame - matches Python: if frame_id > 0: break
                for (int frameId = 0; frameId < Mathf.Min(maxFrames, input.DrivingFrames.Length); frameId++)
                {
                    
                    // Python: img_rgb = frame[:, :, ::-1]  # BGR -> RGB (Unity input is already RGB)
                    var imgRgb = input.DrivingFrames[frameId];
                    
                    // Python: I_p, self.pred_info = predict(frame_id, self.models, x_s_info, R_s, f_s, x_s, img_rgb, self.pred_info)
                    var (Ip, updatedPredInfo) = Predict(frameId, xSInfo, Rs, fs, xs, imgRgb, _predInfo);
                    _predInfo = updatedPredInfo;
                    
                    // Python: if self.flg_composite: driving_img = concat_frame(img_rgb, img_crop_256x256, I_p)
                    // Python: else: driving_img = paste_back(I_p, crop_info["M_c2o"], src_img, mask_ori)
                    Texture2D drivingImg;
                    // generatedFrames.Add(Ip);
                    
                    if (_flgComposite)
                    {
                        drivingImg = ConcatFrame(imgRgb, cropInfo.ImageCrop256x256, Ip);
                    }
                    else
                    {
                        drivingImg = PasteBack(Ip, cropInfo.Transform, srcImgTexture, maskOri);
                    }
                    
                    if (_debugImage != null)
                    {
                        generatedFrames.Add(_debugImage);
                    }                    
                    else if (drivingImg != null)
                    {
                        generatedFrames.Add(drivingImg);
                    }
                }
                
                var result = new LivePortraitResult
                {
                    Success = true,
                    GeneratedFrames = generatedFrames
                };

                var elapsed = start.ElapsedMilliseconds;
                Debug.Log($"[LivePortraitInference] Generation took {elapsed}ms");
                
                return result;
            }
            catch (Exception e)
            {
                Debug.LogError($"[LivePortraitInference] Generation failed: {e.Message}\n{e.StackTrace}");
                return new LivePortraitResult
                {
                    Success = false,
                    ErrorMessage = e.Message,
                    GeneratedFrames = new List<Texture2D>()
                };
            }
        }
        
        /// <summary>
        /// Python: src_preprocess(img) - EXACT MATCH
        /// Returns (byte[] imageData, int width, int height) in RGB24 format
        /// </summary>
        private (byte[], int, int) SrcPreprocess(Texture2D img)
        {
            int h = img.height;
            int w = img.width;
            
            // Get initial image data as byte array (RGB24 format)
            byte[] imageData = GetTextureAsRgb24Bytes(img);
            int currentWidth = w;
            int currentHeight = h;
            
            // Python: max_dim = 1280
            // Python: if max(h, w) > max_dim:
            const int maxDim = 1280;
            if (Mathf.Max(h, w) > maxDim)
            {
                int newHeight, newWidth;
                if (h > w)
                {
                    // Python: new_h = max_dim; new_w = int(w * (max_dim / h))
                    newHeight = maxDim;
                    newWidth = Mathf.RoundToInt(w * ((float)maxDim / h));
                }
                else
                {
                    // Python: new_w = max_dim; new_h = int(h * (max_dim / w))
                    newWidth = maxDim;
                    newHeight = Mathf.RoundToInt(h * ((float)maxDim / w));
                }
                
                // Python: img = cv2.resize(img, (new_w, new_h))
                imageData = ResizeImageBytes(imageData, currentWidth, currentHeight, newWidth, newHeight);
                currentWidth = newWidth;
                currentHeight = newHeight;
            }
            
            // Python: division = 2
            // Python: new_h = img.shape[0] - (img.shape[0] % division)
            // Python: new_w = img.shape[1] - (img.shape[1] % division)
            const int division = 2;
            int finalHeight = currentHeight - (currentHeight % division);
            int finalWidth = currentWidth - (currentWidth % division);
            
            // Python: if new_h == 0 or new_w == 0: return img
            if (finalHeight == 0 || finalWidth == 0)
            {
                return (imageData, currentWidth, currentHeight);
            }
            
            // Python: if new_h != img.shape[0] or new_w != img.shape[1]: img = img[:new_h, :new_w]
            if (finalHeight != currentHeight || finalWidth != currentWidth)
            {
                // Python crops from top-left: img[:new_h, :new_w]
                imageData = CropImageBytes(imageData, currentWidth, currentHeight, finalWidth, finalHeight);
                return (imageData, finalWidth, finalHeight);
            }
            
            return (imageData, currentWidth, currentHeight);
        }
        
        /// <summary>
        /// Convert Texture2D to RGB24 byte array
        /// </summary>
        private byte[] GetTextureAsRgb24Bytes(Texture2D texture)
        {
            var colors = texture.GetPixels();
            var bytes = new byte[colors.Length * 3]; // RGB24 = 3 bytes per pixel
            
            for (int i = 0; i < colors.Length; i++)
            {
                bytes[i * 3] = (byte)(colors[i].r * 255f);     // R
                bytes[i * 3 + 1] = (byte)(colors[i].g * 255f); // G  
                bytes[i * 3 + 2] = (byte)(colors[i].b * 255f); // B
            }
            
            return bytes;
        }
        
        /// <summary>
        /// Resize RGB24 byte array image data using bilinear interpolation
        /// </summary>
        private byte[] ResizeImageBytes(byte[] sourceData, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
        {
            var targetData = new byte[targetWidth * targetHeight * 3];
            
            float xRatio = (float)sourceWidth / targetWidth;
            float yRatio = (float)sourceHeight / targetHeight;
            
            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    float srcX = x * xRatio;
                    float srcY = y * yRatio;
                    
                    int x1 = Mathf.FloorToInt(srcX);
                    int y1 = Mathf.FloorToInt(srcY);
                    int x2 = Mathf.Min(x1 + 1, sourceWidth - 1);
                    int y2 = Mathf.Min(y1 + 1, sourceHeight - 1);
                    
                    float fx = srcX - x1;
                    float fy = srcY - y1;
                    
                    for (int c = 0; c < 3; c++) // RGB channels
                    {
                        int src1 = (y1 * sourceWidth + x1) * 3 + c;
                        int src2 = (y1 * sourceWidth + x2) * 3 + c;
                        int src3 = (y2 * sourceWidth + x1) * 3 + c;
                        int src4 = (y2 * sourceWidth + x2) * 3 + c;
                        
                        float val1 = sourceData[src1] * (1 - fx) + sourceData[src2] * fx;
                        float val2 = sourceData[src3] * (1 - fx) + sourceData[src4] * fx;
                        float finalVal = val1 * (1 - fy) + val2 * fy;
                        
                        int targetIdx = (y * targetWidth + x) * 3 + c;
                        targetData[targetIdx] = (byte)Mathf.Clamp(finalVal, 0, 255);
                    }
                }
            }
            
            return targetData;
        }
        
        /// <summary>
        /// Crop RGB24 byte array from top-left corner (Python-style cropping)
        /// </summary>
        private byte[] CropImageBytes(byte[] sourceData, int sourceWidth, int sourceHeight, int cropWidth, int cropHeight)
        {
            var croppedData = new byte[cropWidth * cropHeight * 3];
            
            for (int y = 0; y < cropHeight; y++)
            {
                for (int x = 0; x < cropWidth; x++)
                {
                    int srcIdx = (y * sourceWidth + x) * 3;
                    int dstIdx = (y * cropWidth + x) * 3;
                    
                    croppedData[dstIdx] = sourceData[srcIdx];         // R
                    croppedData[dstIdx + 1] = sourceData[srcIdx + 1]; // G
                    croppedData[dstIdx + 2] = sourceData[srcIdx + 2]; // B
                }
            }
            
            return croppedData;
        }
        
        /// <summary>
        /// Temporary utility: Convert RGB24 byte array back to Texture2D
        /// TODO: Remove once all methods are converted to work with byte arrays
        /// </summary>
        private Texture2D BytesToTexture2D(byte[] imageData, int width, int height)
        {
            var texture = new Texture2D(width, height, TextureFormat.RGB24, false);
            var colors = new Color[width * height];
            
            for (int i = 0; i < colors.Length; i++)
            {
                colors[i] = new Color(
                    imageData[i * 3] / 255f,     // R
                    imageData[i * 3 + 1] / 255f, // G
                    imageData[i * 3 + 2] / 255f  // B
                );
            }
            
            texture.SetPixels(colors);
            texture.Apply();
            return texture;
        }
        
        /// <summary>
        /// Python: crop_src_image(models, img) - EXACT MATCH
        /// </summary>
        private CropInfo CropSrcImage(Texture2D img)
        {
            
            // Python: face_analysis = models["face_analysis"]
            // Python: src_face = face_analysis(img)
            var srcFaces = FaceAnalysis(img);
            
            // Python: if len(src_face) == 0: print("No face detected in the source image."); return None
            if (srcFaces.Count == 0)
            {
                throw new InvalidOperationException("No face detected in the source image.");
            }
            
            // Python: elif len(src_face) > 1: print(f"More than one face detected in the image, only pick one face.")
            if (srcFaces.Count > 1)
            {
                Debug.LogWarning("More than one face detected in the image, only pick one face.");
            }
            
            // Python: src_face = src_face[0]
            var srcFace = srcFaces[0];
            // Convert Unity Rect to Python format [x1, y1, x2, y2] for logging
            float bx1 = srcFace.BoundingBox.x;
            float by1 = srcFace.BoundingBox.y;
            float bx2 = srcFace.BoundingBox.x + srcFace.BoundingBox.width;
            float by2 = srcFace.BoundingBox.y + srcFace.BoundingBox.height;
            
            // Python: lmk = src_face["landmark_2d_106"]  # this is the 106 landmarks from insightface
            var lmk = srcFace.Landmarks106;
            
            // Python: crop_info = crop_image(img, lmk, dsize=512, scale=2.3, vy_ratio=-0.125)
            var cropInfo = CropImage(img, lmk, 512, 2.3f, -0.125f);            
            
            // Python: lmk = landmark_runner(models, img, lmk)
            lmk = LandmarkRunner(img, lmk);
            
            // Python: crop_info["lmk_crop"] = lmk
            cropInfo.LandmarksCrop = lmk;
            
            // Python: crop_info["img_crop_256x256"] = cv2.resize(crop_info["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            cropInfo.ImageCrop256x256 = ResizeTexture(cropInfo.ImageCrop, 256, 256);
            
            // Python: crop_info["lmk_crop_256x256"] = crop_info["lmk_crop"] * 256 / 512
            cropInfo.LandmarksCrop256x256 = ScaleLandmarks(cropInfo.LandmarksCrop, 256f / 512f);
            
            return cropInfo;
        }
        
        /// <summary>
        /// Python: face_analysis(img) - EXACT MATCH
        /// Implements the complete face detection pipeline from Python
        /// </summary>
        private List<FaceDetectionResult> FaceAnalysis(Texture2D img)
        {
            
            // Python: input_size = 512
            const int inputSize = 512;
            
            // CRITICAL FIX: Match Python's dimension interpretation
            // Python treats image as (height, width, channels) = (img.shape[0], img.shape[1], img.shape[2])
            // Unity texture2D.width/height corresponds to OpenCV width/height
            // So: Python img.shape[0] = height = Unity img.height
            //     Python img.shape[1] = width = Unity img.width
            int pythonHeight = img.height;  // This matches Python's img.shape[0]
            int pythonWidth = img.width;    // This matches Python's img.shape[1]
            
            
            // Python: im_ratio = float(img.shape[0]) / img.shape[1]
            float imRatio = (float)pythonHeight / pythonWidth;
            
            int newHeight, newWidth;
            // Python: if im_ratio > 1: new_height = input_size; new_width = int(new_height / im_ratio)
            if (imRatio > 1)
            {
                newHeight = inputSize;
                newWidth = Mathf.FloorToInt(newHeight / imRatio);
            }
            else
            {
                // Python: else: new_width = input_size; new_height = int(new_width * im_ratio)
                newWidth = inputSize;
                newHeight = Mathf.FloorToInt(newWidth * imRatio);
            }
            
            // Python: det_scale = float(new_height) / img.shape[0]
            float detScale = (float)newHeight / pythonHeight;
            
            // Python: resized_img = cv2.resize(img, (new_width, new_height))
            var resizedImg = ResizeTexture(img, newWidth, newHeight);
            
            // Python: det_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
            // Python: det_img[:new_height, :new_width, :] = resized_img
            var detImg = new Texture2D(inputSize, inputSize, TextureFormat.RGB24, false);
            var detPixels = new Color[inputSize * inputSize];
            var resizedPixels = resizedImg.GetPixels();
            
            // Fill with zeros (black)
            for (int i = 0; i < detPixels.Length; i++)
            {
                detPixels[i] = Color.black;
            }
            
            // Copy resized image to top-left - NO coordinate flipping here since we handle it in tensor creation
            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    int srcIdx = y * newWidth + x;
                    int dstIdx = y * inputSize + x; // Direct copy, no flipping
                    if (srcIdx < resizedPixels.Length)
                    {
                        detPixels[dstIdx] = resizedPixels[srcIdx];
                    }
                }
            }
            
            detImg.SetPixels(detPixels);
            detImg.Apply();
            
            // Python: det_img = (det_img - 127.5) / 128
            // Python: det_img = det_img.transpose(2, 0, 1)  # HWC -> CHW
            // Python: det_img = np.expand_dims(det_img, axis=0)
            // Python: det_img = det_img.astype(np.float32)
            var inputTensor = PreprocessDetectionImage(detImg, inputSize);
            
            // Python: output = det_face.run(None, {"input.1": det_img})
            // Use the actual input name from the model metadata
            string inputName = _detFace.InputMetadata.Keys.First();
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };
            
            
            using var results = _detFace.Run(inputs);
            var outputs = results.ToArray();
            
            
            // Process detection results exactly as in Python
            var faces = ProcessDetectionResults(outputs, detScale);
            
            // Get landmarks for each face
            var finalFaces = new List<FaceDetectionResult>();
            foreach (var face in faces)
            {
                var landmarks = GetLandmark(img, face);
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
            
            // UnityEngine.Object.DestroyImmediate(resizedImg);
            // UnityEngine.Object.DestroyImmediate(detImg);
            
            
            return finalFaces;
        }
        
        /// <summary>
        /// Python: get_landmark(img, face) - EXACT MATCH
        /// </summary>
        private Vector2[] GetLandmark(Texture2D img, FaceDetectionResult face)
        {
            // Python: input_size = 192
            const int inputSize = 192;
            
            // Python: bbox = face["bbox"]
            var bbox = face.BoundingBox;
            
            // Convert Unity Rect (x, y, width, height) to Python format [x1, y1, x2, y2] for logging
            float x1 = bbox.x;
            float y1 = bbox.y; 
            float x2 = bbox.x + bbox.width;
            float y2 = bbox.y + bbox.height;
            
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
            var (alignedImg, transformMatrix) = FaceAlign(img, center, inputSize, scale, rotate);
            
            // Format transform matrix to match Python exactly
            
            // Python: aimg = aimg.transpose(2, 0, 1)  # HWC -> CHW
            // Python: aimg = np.expand_dims(aimg, axis=0)
            // Python: aimg = aimg.astype(np.float32)
            var inputTensor = PreprocessLandmarkImage(alignedImg);
            var tensorData = inputTensor.ToArray();
            // Print the first 10 values of the tensor
            
            // Python: output = landmark.run(None, {"data": aimg})
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("data", inputTensor)
            };
            
            using var results = _landmark2d106.Run(inputs);
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
            var IM = transformMatrix.inverse;// InvertAffineTransformToMatrix(transformMatrix);
            
            landmarks = TransPoints2D(landmarks, IM);
            
            // UnityEngine.Object.DestroyImmediate(alignedImg);
            
            return landmarks;
        }
        
        /// <summary>
        /// Python: landmark_runner(models, img, lmk) - EXACT MATCH
        /// </summary>
        private Vector2[] LandmarkRunner(Texture2D img, Vector2[] lmk)
        {
            // Python: crop_dct = crop_image(img, lmk, dsize=224, scale=1.5, vy_ratio=-0.1)
            var cropDct = CropImage(img, lmk, 224, 1.5f, -0.1f);
            var imgCrop = cropDct.ImageCrop;
            
            // Python: img_crop = img_crop / 255
            // Python: img_crop = img_crop.transpose(2, 0, 1)  # HWC -> CHW
            // Python: img_crop = np.expand_dims(img_crop, axis=0)
            // Python: img_crop = img_crop.astype(np.float32)
            var inputTensor = PreprocessLandmarkRunnerImage(imgCrop);
            
            // Python: net = models["landmark_runner"]
            // Python: output = net.run(None, {"input": img_crop})
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };
            
            using var results = _landmarkRunner.Run(inputs);
            var outputs = results.ToArray();
            
            // Python: out_pts = output[2]
            var outPts = outputs[2].AsTensor<float>().ToArray();
            
            // Python: lmk = out_pts[0].reshape(-1, 2) * 224  # scale to 0-224
            var refinedLmk = new Vector2[outPts.Length / 2];
            for (int i = 0; i < refinedLmk.Length; i++)
            {
                refinedLmk[i] = new Vector2(outPts[i * 2] * 224f, outPts[i * 2 + 1] * 224f);
            }
            
            // Python: M = crop_dct["M_c2o"]
            // Python: lmk = lmk @ M[:2, :2].T + M[:2, 2]
            refinedLmk = TransformLandmarksWithMatrix(refinedLmk, cropDct.Transform);
            
            // UnityEngine.Object.DestroyImmediate(imgCrop);
            
            return refinedLmk;
        }
        
        /// <summary>
        /// Python: preprocess(img) - EXACT MATCH
        /// </summary>
        private float[] Preprocess(Texture2D img)
        {
            // Python: img = img / 255.0
            // Python: img = np.clip(img, 0, 1)  # clip to 0~1
            // Python: img = img.transpose(2, 0, 1)  # HxWx3x1 -> 1x3xHxW
            // Python: img = np.expand_dims(img, axis=0)
            // Python: img = img.astype(np.float32)
            
            
            var pixels = img.GetPixels();
            int height = img.height;
            int width = img.width;
            
            // Calculate input range
            float minVal = float.MaxValue, maxVal = float.MinValue;
            foreach (var pixel in pixels)
            {
                minVal = Mathf.Min(minVal, Mathf.Min(pixel.r, Mathf.Min(pixel.g, pixel.b)));
                maxVal = Mathf.Max(maxVal, Mathf.Max(pixel.r, Mathf.Max(pixel.g, pixel.b)));
            }
            
            var data = new float[1 * 3 * height * width];
            
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        // CRITICAL: Unity GetPixels() is bottom-left origin, flip Y for ONNX (top-left)
                        int unityY = height - 1 - h; // Flip Y coordinate for ONNX coordinate system
                        int unityIdx = unityY * width + w;
                        int outputIdx = c * height * width + h * width + w;
                        
                        float value = c == 0 ? pixels[unityIdx].r : c == 1 ? pixels[unityIdx].g : pixels[unityIdx].b;
                        data[outputIdx] = Mathf.Clamp01(value); // Normalize and clip to [0,1]
                    }
                }
            }
            
            float dataMin = data.Min(), dataMax = data.Max();
            
            return data;
        }
        
        /// <summary>
        /// Python: get_kp_info(models, x) - EXACT MATCH
        /// </summary>
        private MotionInfo GetKpInfo(float[] preprocessedData)
        {
            float dataMin = preprocessedData.Min(), dataMax = preprocessedData.Max();
            
            // Convert to tensor
            var inputTensor = new DenseTensor<float>(preprocessedData, new[] { 1, 3, 256, 256 });
            
            // Python: net = models["motion_extractor"]
            // Python: output = net.run(None, {"img": x})
            // Use the actual input name from the model metadata
            string inputName = _motionExtractor.InputMetadata.Keys.First();
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };
            
            using var results = _motionExtractor.Run(inputs);
            var outputs = results.ToArray();
            
            // Python: pitch, yaw, roll, t, exp, scale, kp = output
            var pitch = outputs[0].AsTensor<float>().ToArray();
            var yaw = outputs[1].AsTensor<float>().ToArray();
            var roll = outputs[2].AsTensor<float>().ToArray();
            var t = outputs[3].AsTensor<float>().ToArray();
            var exp = outputs[4].AsTensor<float>().ToArray();
            var scale = outputs[5].AsTensor<float>().ToArray();
            var kp = outputs[6].AsTensor<float>().ToArray();
            
            
            // Python: pred = softmax(kp_info["pitch"], axis=1)
            // Python: degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
            // Python: kp_info["pitch"] = degree[:, None]  # Bx1
            var processedPitch = ProcessAngleSoftmax(pitch);
            var processedYaw = ProcessAngleSoftmax(yaw);
            var processedRoll = ProcessAngleSoftmax(roll);
            
            
            // Python: bs = kp_info["kp"].shape[0]
            // Python: kp_info["kp"] = kp_info["kp"].reshape(bs, -1, 3)  # BxNx3
            // Python: kp_info["exp"] = kp_info["exp"].reshape(bs, -1, 3)  # BxNx3
            
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
        
        /// <summary>
        /// Python: extract_feature_3d(models, x) - EXACT MATCH
        /// </summary>
        private float[] ExtractFeature3d(float[] preprocessedData)
        {
            float dataMin = preprocessedData.Min(), dataMax = preprocessedData.Max();
            
            var inputTensor = new DenseTensor<float>(preprocessedData, new[] { 1, 3, 256, 256 });
            
            // Python: net = models["appearance_feature_extractor"]
            // Python: output = net.run(None, {"img": x})
            // Use the actual input name from the model metadata
            string inputName = _appearanceFeatureExtractor.InputMetadata.Keys.First();
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };
            
            using var results = _appearanceFeatureExtractor.Run(inputs);
            var output = results.First().AsTensor<float>().ToArray();
            
            var outputTensor = results.First().AsTensor<float>();
            var outputShape = outputTensor.Dimensions.ToArray();
            float outputMin = output.Min(), outputMax = output.Max();
            
            // Python: f_s = output[0]
            // Python: f_s = f_s.astype(np.float32)
            return output;
        }
        
        /// <summary>
        /// Python: transform_keypoint(x_s_info) - EXACT MATCH
        /// Transform the implicit keypoints with the pose, shift, and expression deformation
        /// kp: BxNx3
        /// </summary>
        private float[] TransformKeypoint(MotionInfo xSInfo)
        {
            // Python: kp = kp_info["kp"]  # (bs, k, 3)
            var kp = xSInfo.Keypoints;
            
            // Python: pitch, yaw, roll = kp_info["pitch"], kp_info["yaw"], kp_info["roll"]
            var pitch = xSInfo.Pitch;
            var yaw = xSInfo.Yaw;
            var roll = xSInfo.Roll;
            
            // Python: t, exp = kp_info["t"], kp_info["exp"]
            var t = xSInfo.Translation;
            var exp = xSInfo.Expression;
            
            // Python: scale = kp_info["scale"]
            var scale = xSInfo.Scale;
            
            // Python: bs = kp.shape[0]
            // Python: num_kp = kp.shape[1]  # Bxnum_kpx3
            // int bs = 1; // Batch size is always 1 in our case
            int numKp = kp.Length / 3;
            
            // Python: rot_mat = get_rotation_matrix(pitch, yaw, roll)  # (bs, 3, 3)
            var rotMat = GetRotationMatrix(pitch, yaw, roll);
            
            // Python: kp_transformed = kp.reshape(bs, num_kp, 3) @ rot_mat + exp.reshape(bs, num_kp, 3)
            var kpTransformed = new float[kp.Length];
            
            for (int i = 0; i < numKp; i++)
            {
                // Get keypoint coordinates
                float x = kp[i * 3 + 0];
                float y = kp[i * 3 + 1];
                float z = kp[i * 3 + 2];
                
                // Matrix multiplication: kp @ rot_mat
                float newX = x * rotMat[0, 0] + y * rotMat[1, 0] + z * rotMat[2, 0];
                float newY = x * rotMat[0, 1] + y * rotMat[1, 1] + z * rotMat[2, 1];
                float newZ = x * rotMat[0, 2] + y * rotMat[1, 2] + z * rotMat[2, 2];
                
                // Add expression deformation: + exp.reshape(bs, num_kp, 3)
                if (i * 3 + 2 < exp.Length)
                {
                    newX += exp[i * 3 + 0];
                    newY += exp[i * 3 + 1];
                    newZ += exp[i * 3 + 2];
                }
                
                kpTransformed[i * 3 + 0] = newX;
                kpTransformed[i * 3 + 1] = newY;
                kpTransformed[i * 3 + 2] = newZ;
            }
            
            // Python: kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
            if (scale.Length > 0)
            {
                float scaleValue = scale[0];
                for (int i = 0; i < kpTransformed.Length; i++)
                {
                    kpTransformed[i] *= scaleValue;
                }
            }
            
            // Python: kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty
            if (t.Length >= 2)
            {
                for (int i = 0; i < numKp; i++)
                {
                    kpTransformed[i * 3 + 0] += t[0]; // tx
                    kpTransformed[i * 3 + 1] += t[1]; // ty
                    // Don't add tz to z coordinate as per Python comment "remove z"
                }
            }
            
            return kpTransformed;
        }
        
        /// <summary>
        /// Python: predict(frame_id, models, x_s_info, R_s, f_s, x_s, img, pred_info) - EXACT MATCH
        /// </summary>
        private (Texture2D, LivePortraitPredInfo) Predict(int frameId, MotionInfo xSInfo, float[,] Rs, float[] fs, float[] xs, 
            Texture2D img, LivePortraitPredInfo predInfo)
        {
            
            // Python: frame_0 = pred_info['lmk'] is None
            bool frame0 = predInfo.Landmarks == null;
            
            Vector2[] lmk;
            if (frame0)
            {
                // Python: face_analysis = models["face_analysis"]
                // Python: src_face = face_analysis(img)
                var srcFaces = FaceAnalysis(img);
                if (srcFaces.Count == 0)
                {
                    throw new InvalidOperationException("No face detected in the frame");
                }
                
                if (srcFaces.Count > 1)
                {
                    // Debug.LogWarning("More than one face detected in the driving frame, only pick one face.");
                }
                
                // Python: src_face = src_face[0]
                // Python: lmk = src_face["landmark_2d_106"]
                var srcFace = srcFaces[0];
                lmk = srcFace.Landmarks106;
                
                // Python: lmk = landmark_runner(models, img, lmk)
                lmk = LandmarkRunner(img, lmk);
            }
            else
            {
                // Python: lmk = landmark_runner(models, img, pred_info['lmk'])
                lmk = LandmarkRunner(img, predInfo.Landmarks);
            }
            
            // Python: pred_info['lmk'] = lmk
            predInfo.Landmarks = lmk;
            
            // Python: calc_driving_ratio - CRITICAL FIX: Now implementing the missing calculation
            // Python: lmk = lmk[None]  # Add batch dimension - CRITICAL: This changes shape from (106,2) to (1,106,2)
            // IMPORTANT: Python adds batch dimension here, but our CalculateDistanceRatio function expects unbatched landmarks
            // So we pass the original lmk array directly since our function handles single batch internally
            
            // Python: c_d_eyes = np.concatenate([calculate_distance_ratio(lmk, 6, 18, 0, 12), calculate_distance_ratio(lmk, 30, 42, 24, 36)], axis=1)
            // Python: c_d_lip = calculate_distance_ratio(lmk, 90, 102, 48, 66)
            // Python: c_d_eyes = c_d_eyes.astype(np.float32)
            // Python: c_d_lip = c_d_lip.astype(np.float32)
            
            var cDEyes1 = CalculateDistanceRatio(lmk, 6, 18, 0, 12);
            var cDEyes2 = CalculateDistanceRatio(lmk, 30, 42, 24, 36);
            // Python concatenates these along axis=1
            var cDEyes = new float[cDEyes1.Length + cDEyes2.Length];
            Array.Copy(cDEyes1, 0, cDEyes, 0, cDEyes1.Length);
            Array.Copy(cDEyes2, 0, cDEyes, cDEyes1.Length, cDEyes2.Length);
            
            var cDLip = CalculateDistanceRatio(lmk, 90, 102, 48, 66);
            
            // Convert to float32 (already float in C#)
            // Note: These values are computed but never used in ONNX inference, matching Python behavior exactly
            
            // Python: prepare_driving_videos
            // Python: img = cv2.resize(img, (256, 256))
            var img256 = ResizeTexture(img, 256, 256);
            
            // Python: I_d = preprocess(img)
            var Id = Preprocess(img256);
            
            // Python: collect s_d, R_d, δ_d and t_d for inference
            // Python: x_d_info = get_kp_info(models, I_d)
            var xDInfo = GetKpInfo(Id);
            
            // Python: R_d = get_rotation_matrix(x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"])
            var Rd = GetRotationMatrix(xDInfo.Pitch, xDInfo.Yaw, xDInfo.Roll);
            
            // CRITICAL FIX: Python restructures x_d_info to only contain specific fields with explicit float32 conversion
            // Python: x_d_info = {
            //     "scale": x_d_info["scale"].astype(np.float32),
            //     "R_d": R_d.astype(np.float32),
            //     "exp": x_d_info["exp"].astype(np.float32),
            //     "t": x_d_info["t"].astype(np.float32),
            // }
            
            // Ensure Rd is float32 equivalent (Python: R_d.astype(np.float32))
            var RdFloat32 = EnsureFloat32Matrix(Rd);
            
            // Restructure xDInfo to match Python exactly - only keep the fields Python keeps
            var restructuredXDInfo = new MotionInfo
            {
                Scale = EnsureFloat32Array(xDInfo.Scale),
                Expression = EnsureFloat32Array(xDInfo.Expression),
                Translation = EnsureFloat32Array(xDInfo.Translation),
                RotationMatrix = RdFloat32,  // CRITICAL: Store R_d in restructured info as Python does
                // Python doesn't keep pitch, yaw, roll, keypoints in the restructured version
            };
            
            // Use restructured version for the rest of the function
            xDInfo = restructuredXDInfo;
            Rd = RdFloat32;
            
            if (frame0)
            {
                // Python: pred_info['x_d_0_info'] = x_d_info
                predInfo.InitialMotionInfo = xDInfo;
            }
            
            // Python: x_d_0_info = pred_info['x_d_0_info']
            var xD0Info = predInfo.InitialMotionInfo;
            
            // Python: R_d_0 = x_d_0_info["R_d"]
            var Rd0 = xD0Info.RotationMatrix;  // FIXED: Access stored rotation matrix directly
            
            // Python: R_new = (R_d @ R_d_0.transpose(0, 2, 1)) @ R_s
            // CRITICAL FIX: Python transpose(0, 2, 1) swaps last two dimensions for batch matrices
            // For 3x3 matrices, this is equivalent to standard matrix transpose
            var Rd0Transposed = TransposeMatrix(Rd0);
            var RdTimesRd0T = MatrixMultiply(Rd, Rd0Transposed);
            var RNew = MatrixMultiply(RdTimesRd0T, Rs);
            
            // Python: delta_new = x_s_info["exp"] + (x_d_info["exp"] - x_d_0_info["exp"])
            var expDiff = SubtractArrays(xDInfo.Expression, xD0Info.Expression);
            var deltaNew = AddArrays(xSInfo.Expression, expDiff);
            
            // Python: scale_new = x_s_info["scale"] * (x_d_info["scale"] / x_d_0_info["scale"])
            var scaleDiff = DivideArrays(xDInfo.Scale, xD0Info.Scale);
            var scaleNew = MultiplyArrays(xSInfo.Scale, scaleDiff);
            
            // Python: t_new = x_s_info["t"] + (x_d_info["t"] - x_d_0_info["t"])
            var tDiff = SubtractArrays(xDInfo.Translation, xD0Info.Translation);
            var tNew = AddArrays(xSInfo.Translation, tDiff);
            
            // Python: t_new[..., 2] = 0  # zero tz
            if (tNew.Length >= 3) tNew[2] = 0;
            
            // Python: x_c_s = x_s_info["kp"]
            var xCs = xSInfo.Keypoints;
            
            // Python: x_d_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
            var xDNew = CalculateNewKeypoints(xCs, RNew, deltaNew, scaleNew, tNew);
            
            // Debug: Check keypoint transformation values
            
            // Python: x_d_new = stitching(models, x_s, x_d_new)
            xDNew = Stitching(xs, xDNew);
            
            // Python: out = warping_spade(models, f_s, x_s, x_d_new)
            var output = WarpingSpade(fs, xs, xDNew);
            
            float outputMin = output.Min(), outputMax = output.Max();
            
            // Python: out = out.transpose(0, 2, 3, 1)  # 1x3xHxW -> 1xHxWx3
            // Python: out = np.clip(out, 0, 1)  # clip to 0~1
            // Python: out = (out * 255).astype(np.uint8)  # 0~1 -> 0~255
            // Python: I_p = out[0]
            var resultTexture = ConvertOutputToTexture(output);
            
            
            // UnityEngine.Object.DestroyImmediate(img256);
            
            // Python: return I_p, pred_info
            return (resultTexture, predInfo);
        }
        
        /// <summary>
        /// Python: stitching(models, kp_source, kp_driving) - EXACT MATCH
        /// </summary>
        private float[] Stitching(float[] kpSource, float[] kpDriving)
        {
            // Python: bs, num_kp = kp_source.shape[:2]
            // Python: kp_driving_new = kp_driving
            var kpDrivingNew = new float[kpDriving.Length];
            Array.Copy(kpDriving, kpDrivingNew, kpDriving.Length);
            
            // Python: bs_src = kp_source.shape[0]
            // Python: bs_dri = kp_driving.shape[0]
            // Python: feat = np.concatenate([kp_source.reshape(bs_src, -1), kp_driving.reshape(bs_dri, -1)], axis=1)
            var feat = new float[kpSource.Length + kpDriving.Length];
            Array.Copy(kpSource, 0, feat, 0, kpSource.Length);
            Array.Copy(kpDriving, 0, feat, kpSource.Length, kpDriving.Length);
            
            var inputTensor = new DenseTensor<float>(feat, new[] { 1, feat.Length });
            
            // Python: net = models["stitching"]
            // Python: output = net.run(None, {"input": feat})
            // Use actual input name from model metadata
            string inputName = _stitching.InputMetadata.Keys.First();
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };
            
            using var results = _stitching.Run(inputs);
            var delta = results.First().AsTensor<float>().ToArray();
            
            // Python: delta_exp = delta[..., : 3 * num_kp].reshape(bs, num_kp, 3)  # 1x20x3
            // Python: delta_tx_ty = delta[..., 3 * num_kp : 3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2
            int numKp = kpDriving.Length / 3;
            
            // Python: kp_driving_new += delta_exp
            for (int i = 0; i < numKp * 3 && i < delta.Length; i++)
            {
                kpDrivingNew[i] += delta[i];
            }
            
            // Python: kp_driving_new[..., :2] += delta_tx_ty
            if (delta.Length >= numKp * 3 + 2)
            {
                float deltaX = delta[numKp * 3];
                float deltaY = delta[numKp * 3 + 1];
                
                for (int i = 0; i < numKp; i++)
                {
                    kpDrivingNew[i * 3] += deltaX;     // x coordinate
                    kpDrivingNew[i * 3 + 1] += deltaY; // y coordinate
                }
            }
            
            return kpDrivingNew;
        }
        
        /// <summary>
        /// Python: warping_spade(models, feature_3d, kp_source, kp_driving) - EXACT MATCH
        /// </summary>
        private float[] WarpingSpade(float[] feature3d, float[] kpSource, float[] kpDriving)
        {
            // CRITICAL FIX: Verify tensor shapes match Python exactly
            // Python: feature_3d shape should be (1, 32, 16, 64, 64) = 2,097,152 elements
            // Python: kp_source shape should be (1, 21, 3) = 63 elements  
            // Python: kp_driving shape should be (1, 21, 3) = 63 elements
            
            
            // Verify expected sizes
            int expectedFeature3DSize = 1 * 32 * 16 * 64 * 64; // 2,097,152
            int expectedKpSize = 21 * 3; // 63 (21 keypoints * 3 coordinates)
            
            if (feature3d.Length != expectedFeature3DSize)
            {
                Debug.LogError($"[DEBUG_WARPING_SPADE] Feature3D size mismatch! Expected: {expectedFeature3DSize}, Got: {feature3d.Length}");
            }
            
            if (kpSource.Length != expectedKpSize || kpDriving.Length != expectedKpSize)
            {
                Debug.LogError($"[DEBUG_WARPING_SPADE] Keypoint size mismatch! Expected: {expectedKpSize}, Got kpSource: {kpSource.Length}, kpDriving: {kpDriving.Length}");
            }
            
            // Create tensors with proper shapes
            var feature3DTensor = new DenseTensor<float>(feature3d, new[] { 1, 32, 16, 64, 64 });
            var kpSourceTensor = new DenseTensor<float>(kpSource, new[] { 1, kpSource.Length / 3, 3 });
            var kpDrivingTensor = new DenseTensor<float>(kpDriving, new[] { 1, kpDriving.Length / 3, 3 });
            
            
            // Python: net = models["warping_spade"]
            // Python: output = net.run(None, {"feature_3d": feature_3d, "kp_driving": kp_driving, "kp_source": kp_source})
            // Use actual input names from model metadata
            var inputNames = _warpingSpade.InputMetadata.Keys.ToArray();
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputNames[0], feature3DTensor),  // feature_3d
                NamedOnnxValue.CreateFromTensor(inputNames[1], kpDrivingTensor), // kp_driving  
                NamedOnnxValue.CreateFromTensor(inputNames[2], kpSourceTensor)   // kp_source
            };
            
            using var results = _warpingSpade.Run(inputs);
            var outputs = results.ToArray();
            
            var outputShape = outputs[0].AsTensor<float>().Dimensions.ToArray();
            
            // Python: return output[0] - take the first output (warped_feature)
            var output = outputs[0].AsTensor<float>().ToArray();
            
            
            return output;
        }
        
        // Helper methods - all implemented inline for self-sufficiency
        private Texture2D ResizeTexture(Texture2D source, int width, int height)
        {
            var result = new Texture2D(width, height, TextureFormat.RGB24, false);
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
            for (int i = 0; i < lmk.Length; i++)
            {
                // proved correct
            }
            // Python: crop_image(img, pts: np.ndarray, dsize=224, scale=1.5, vy_ratio=-0.1) - EXACT MATCH
            var (MInv, _) = EstimateSimilarTransformFromPts(lmk, dsize, scale, 0f, vyRatio, true);
            
            var imgCrop = TransformImgExact(img, MInv, dsize);
            var ptCrop = TransformPts(lmk, MInv);
            
            // Python: M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=np.float32)])
            var Mo2c = Matrix4x4.identity;
            Mo2c.m00 = MInv[0, 0]; Mo2c.m01 = MInv[0, 1]; Mo2c.m03 = MInv[0, 2];
            Mo2c.m10 = MInv[1, 0]; Mo2c.m11 = MInv[1, 1]; Mo2c.m13 = MInv[1, 2];
            Mo2c.m20 = 0f; Mo2c.m21 = 0f; Mo2c.m22 = 1f; Mo2c.m23 = 0f;
            Mo2c.m30 = 0f; Mo2c.m31 = 0f; Mo2c.m32 = 0f; Mo2c.m33 = 1f;
            
            // Python: M_c2o = np.linalg.inv(M_o2c)
            var Mc2o = Mo2c.inverse;
            
            var cropInfo = new CropInfo
            {
                ImageCrop = imgCrop,
                Transform = Mc2o,
                InverseTransform = Mo2c,
                LandmarksCrop = ptCrop
            };
            
            return cropInfo;
        }
        
        /// <summary>
        /// Python: softmax processing for angle predictions - EXACT MATCH
        /// </summary>
        private float[] ProcessAngleSoftmax(float[] angleLogits)
        {
            // Python: pred = softmax(kp_info["pitch"], axis=1)
            var softmaxValues = Softmax(angleLogits);
            
            // Python: degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
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
        
        /// <summary>
        /// Python: get_rotation_matrix(pitch, yaw, roll) - EXACT MATCH
        /// </summary>
        private float[,] GetRotationMatrix(float[] pitch, float[] yaw, float[] roll)
        {
            // Python: pitch, yaw, roll are Bx1 arrays. Here they are float[] of length 1.
            float p = pitch[0] * Mathf.Deg2Rad;
            float y = yaw[0] * Mathf.Deg2Rad;
            float r = roll[0] * Mathf.Deg2Rad;
            
            // Python: x, y, z = pitch, yaw, roll
            float cos_p = Mathf.Cos(p);
            float sin_p = Mathf.Sin(p);
            float cos_y = Mathf.Cos(y);
            float sin_y = Mathf.Sin(y);
            float cos_r = Mathf.Cos(r);
            float sin_r = Mathf.Sin(r);
            
            // Python: rot_x
            var rotX = new float[3, 3] {
                { 1, 0, 0 },
                { 0, cos_p, -sin_p },
                { 0, sin_p, cos_p }
            };
            
            // Python: rot_y
            var rotY = new float[3, 3] {
                { cos_y, 0, sin_y },
                { 0, 1, 0 },
                { -sin_y, 0, cos_y }
            };

            // Python: rot_z
            var rotZ = new float[3, 3] {
                { cos_r, -sin_r, 0 },
                { sin_r, cos_r, 0 },
                { 0, 0, 1 }
            };
            
            // Python: rot = rot_z @ rot_y @ rot_x
            var rotZY = MatrixMultiply(rotZ, rotY);
            var rot = MatrixMultiply(rotZY, rotX);
            
            // Python: return rot.transpose(0, 2, 1)
            return TransposeMatrix(rot);
        }
        
        /// <summary>
        /// Python: parse_pt2_from_pt106() - EXACT MATCH
        /// Parsing the 2 points according to the 106 points
        /// </summary>
        private Vector2[] ParsePt2FromPt106(Vector2[] pt106, bool useLip)
        {
            // Python: pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)
            Vector2 ptLeftEye = (pt106[33] + pt106[35] + pt106[40] + pt106[39]) / 4f;
            
            // Python: pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)
            Vector2 ptRightEye = (pt106[87] + pt106[89] + pt106[94] + pt106[93]) / 4f;
            
            Vector2[] pt2;
            
            if (useLip)
            {
                // Python: pt_center_eye = (pt_left_eye + pt_right_eye) / 2
                Vector2 ptCenterEye = (ptLeftEye + ptRightEye) / 2f;
                
                // Python: pt_center_lip = (pt106[52] + pt106[61]) / 2
                Vector2 ptCenterLip = (pt106[52] + pt106[61]) / 2f;
                
                // Python: pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
                pt2 = new Vector2[] { ptCenterEye, ptCenterLip };
            }
            else
            {
                // Python: pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
                pt2 = new Vector2[] { ptLeftEye, ptRightEye };
            }
            
            return pt2;
        }
        
        /// <summary>
        /// Python: parse_pt2_from_pt_x() - EXACT MATCH
        /// </summary>
        private Vector2[] ParsePt2FromPtX(Vector2[] pts, bool useLip)
        {
            var pt2 = ParsePt2FromPt106(pts, useLip);
            
            if (!useLip)
            {
                // Python: v = pt2[1] - pt2[0]
                // Python: pt2[1, 0] = pt2[0, 0] - v[1]
                // Python: pt2[1, 1] = pt2[0, 1] + v[0]
                Vector2 v = pt2[1] - pt2[0];
                pt2[1] = new Vector2(pt2[0].x - v.y, pt2[0].y + v.x);
            }
            
            return pt2;
        }
        
        /// <summary>
        /// Python: parse_rect_from_landmark() - EXACT MATCH
        /// Parsing center, size, angle from landmarks
        /// </summary>
        private (Vector2, Vector2, float) ParseRectFromLandmark(Vector2[] pts, float scale, bool needSquare, float vxRatio, float vyRatio, bool useDegFlag)
        {
            var pt2 = ParsePt2FromPtX(pts, true);  // use_lip=True
            
            // Python: uy = pt2[1] - pt2[0]
            Vector2 uy = pt2[1] - pt2[0];
            float l = uy.magnitude;
            
            // Python: if l <= 1e-3: uy = np.array([0, 1], dtype=np.float32)
            if (l <= 1e-3f)
            {
                uy = new Vector2(0f, 1f);
            }
            else
            {
                uy /= l;  // Python: uy /= l
            }
            
            // Python: ux = np.array((uy[1], -uy[0]), dtype=np.float32)
            Vector2 ux = new(uy.y, -uy.x);

            
            // Python: angle = acos(ux[0])
            // Python: if ux[1] < 0: angle = -angle
            float angle = Mathf.Acos(ux.x);
            // float angle = Mathf.Acos(Mathf.Clamp(ux.x, -1f, 1f));
            if (ux.y < 0)
            {
                angle = -angle;
            }
            
            // Python: M = np.array([ux, uy])
            float[,] M = new float[,] { { ux.x, ux.y }, { uy.x, uy.y } };
            
            // Python: center0 = np.mean(pts, axis=0)
            Vector2 center0 = Vector2.zero;
            for (int i = 0; i < pts.Length; i++)
            {
                center0 += pts[i];
            }
            center0 /= pts.Length;
            
            // Python: rpts = (pts - center0) @ M.T
            Vector2[] rpts = new Vector2[pts.Length];
            for (int i = 0; i < pts.Length; i++)
            {
                Vector2 centered = pts[i] - center0;
                rpts[i] = new Vector2(
                    centered.x * M[0, 0] + centered.y * M[1, 0],  // @ M.T means transpose
                    centered.x * M[0, 1] + centered.y * M[1, 1]
                );
            }
            
            // Python: lt_pt = np.min(rpts, axis=0)
            // Python: rb_pt = np.max(rpts, axis=0)
            Vector2 ltPt = new(float.MaxValue, float.MaxValue);
            Vector2 rbPt = new(float.MinValue, float.MinValue);
            
            for (int i = 0; i < rpts.Length; i++)
            {
                if (rpts[i].x < ltPt.x) ltPt.x = rpts[i].x;
                if (rpts[i].y < ltPt.y) ltPt.y = rpts[i].y;
                if (rpts[i].x > rbPt.x) rbPt.x = rpts[i].x;
                if (rpts[i].y > rbPt.y) rbPt.y = rpts[i].y;
            }
            
            // Python: center1 = (lt_pt + rb_pt) / 2
            Vector2 center1 = (ltPt + rbPt) / 2f;
            
            // Python: size = rb_pt - lt_pt
            Vector2 size = rbPt - ltPt;
            
            // Python: if need_square: m = max(size[0], size[1]); size[0] = m; size[1] = m
            if (needSquare)
            {
                float m = Mathf.Max(size.x, size.y);
                size.x = m;
                size.y = m;
            }
            
            // Python: size *= scale
            size *= scale;
            
            // Python: center = center0 + ux * center1[0] + uy * center1[1]
            Vector2 center = center0 + ux * center1.x + uy * center1.y;
            
            // Python: center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
            center = center + ux * (vxRatio * size.x) + uy * (vyRatio * size.y);
            
            // Python: if use_deg_flag: angle = degrees(angle)
            if (useDegFlag)
            {
                angle *= Mathf.Rad2Deg;
            }
            
            return (center, size, angle);
        }
        
        /// <summary>
        /// Python: _estimate_similar_transform_from_pts() - EXACT MATCH
        /// Calculate the affine matrix of the cropped image from sparse points
        /// </summary>
        private (float[,], float[,]) EstimateSimilarTransformFromPts(Vector2[] pts, int dsize, float scale, float vxRatio, float vyRatio, bool flagDoRot)
        {
            var (center, size, angle) = ParseRectFromLandmark(pts, scale, true, vxRatio, vyRatio, false);

            
            float s = dsize / size.x;  // Python: s = dsize / size[0]
            Vector2 tgtCenter = new(dsize / 2f, dsize / 2f);  // Python: tgt_center = np.array([dsize / 2, dsize / 2])
            
            float[,] MInv;
            
            if (flagDoRot)
            {
                // Python: costheta, sintheta = cos(angle), sin(angle)
                float costheta = Mathf.Cos(angle);
                float sintheta = Mathf.Sin(angle);
                float cx = center.x, cy = center.y;  // Python: cx, cy = center[0], center[1]
                float tcx = tgtCenter.x, tcy = tgtCenter.y;  // Python: tcx, tcy = tgt_center[0], tgt_center[1]
                
                // Python: M_INV = np.array([[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
                //                          [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]])
                MInv = new float[,] {
                    { s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy) },
                    { -s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy) }
                };
            }
            else
            {
                // Python: M_INV = np.array([[s, 0, tgt_center[0] - s * center[0]],
                //                          [0, s, tgt_center[1] - s * center[1]]])
                MInv = new float[,] {
                    { s, 0, tgtCenter.x - s * center.x },
                    { 0, s, tgtCenter.y - s * center.y }
                };
            }
            
            // Python: M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
            // Python: M = np.linalg.inv(M_INV_H)
            var MInvH = new float[3, 3] {
                { MInv[0, 0], MInv[0, 1], MInv[0, 2] },
                { MInv[1, 0], MInv[1, 1], MInv[1, 2] },
                { 0f, 0f, 1f }
            };
            
            var M = InvertMatrix3x3(MInvH);
            var M2x3 = new float[,] {
                { M[0, 0], M[0, 1], M[0, 2] },
                { M[1, 0], M[1, 1], M[1, 2] }
            };

            // Python: return M_INV, M[:2, ...]
            return (MInv, M2x3);
        }
        
        /// <summary>
        /// Matrix operations matching Python numpy - EXACT MATCH
        /// </summary>
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
        
        /// <summary>
        /// Python: _transform_pts() - EXACT MATCH
        /// Conduct similarity or affine transformation to the pts
        /// </summary>
        private Vector2[] TransformPts(Vector2[] pts, float[,] M)
        {
            // Python: return pts @ M[:2, :2].T + M[:2, 2]
            var result = new Vector2[pts.Length];
            
            for (int i = 0; i < pts.Length; i++)
            {
                result[i] = new Vector2(
                    pts[i].x * M[0, 0] + pts[i].y * M[0, 1] + M[0, 2],
                    pts[i].x * M[1, 0] + pts[i].y * M[1, 1] + M[1, 2]
                );
            }
            
            return result;
        }
        
        /// <summary>
        /// Python: distance2bbox() - EXACT MATCH
        /// </summary>
        private float[] Distance2Bbox(float[,] points, float[] distance)
        {
            int numPoints = points.GetLength(0);
            var result = new float[numPoints * 4];
            
            for (int i = 0; i < numPoints; i++)
            {
                float x = points[i, 0];
                float y = points[i, 1];
                
                // Python: x1 = points[:, 0] - distance[:, 0]
                // Python: y1 = points[:, 1] - distance[:, 1]
                // Python: x2 = points[:, 0] + distance[:, 2]
                // Python: y2 = points[:, 1] + distance[:, 3]
                result[i * 4 + 0] = x - distance[i * 4 + 0];  // x1
                result[i * 4 + 1] = y - distance[i * 4 + 1];  // y1
                result[i * 4 + 2] = x + distance[i * 4 + 2];  // x2
                result[i * 4 + 3] = y + distance[i * 4 + 3];  // y2
            }
            
            return result;
        }
        
        /// <summary>
        /// Python: distance2kps() - EXACT MATCH
        /// </summary>
        private float[] Distance2Kps(float[,] points, float[] distance)
        {
            int numPoints = points.GetLength(0);
            var result = new float[numPoints * 10]; // 5 keypoints * 2 coords
            
            for (int i = 0; i < numPoints; i++)
            {
                float x = points[i, 0];
                float y = points[i, 1];
                
                // Python: for i in range(0, distance.shape[1], 2):
                //             px = points[:, i % 2] + distance[:, i]
                //             py = points[:, i % 2 + 1] + distance[:, i + 1]
                for (int kp = 0; kp < 5; kp++) // 5 keypoints
                {
                    int distIdx = kp * 2;
                    result[i * 10 + kp * 2 + 0] = x + distance[i * 10 + distIdx + 0];     // px
                    result[i * 10 + kp * 2 + 1] = y + distance[i * 10 + distIdx + 1];     // py
                }
            }
            
            return result;
        }
        
        /// <summary>
        /// Python: nms_boxes() - EXACT MATCH
        /// </summary>
        private List<int> NmsBoxes(List<float[]> boxes, List<float> scores, float iouThreshold)
        {
            if (boxes.Count == 0)
            {
                return new List<int>();
            }

            var keep = new List<bool>();

            for (int i = 0; i < boxes.Count; i++)
            {
                bool isKeep = true;
                for (int j = 0; j < i; j++)
                {
                    if (!keep[j])
                    {
                        continue;
                    }

                    float iou = BbIntersectionOverUnion(boxes[i], boxes[j]);
                    if (iou >= iouThreshold)
                    {
                        if (scores[i] > scores[j])
                        {
                            keep[j] = false;
                        }
                        else
                        {
                            isKeep = false;
                            break;
                        }
                    }
                }
                keep.Add(isKeep);
            }

            var keepIndices = new List<int>();
            for (int i = 0; i < keep.Count; i++)
            {
                if (keep[i])
                {
                    keepIndices.Add(i);
                }
            }
            return keepIndices;
        }
        
        /// <summary>
        /// Python: bb_intersection_over_union() - EXACT MATCH
        /// </summary>
        private float BbIntersectionOverUnion(float[] boxA, float[] boxB)
        {
            // Python: xA = max(boxA[0], boxB[0])
            // Python: yA = max(boxA[1], boxB[1])
            // Python: xB = min(boxA[2], boxB[2])
            // Python: yB = min(boxA[3], boxB[3])
            float xA = Mathf.Max(boxA[0], boxB[0]);
            float yA = Mathf.Max(boxA[1], boxB[1]);
            float xB = Mathf.Min(boxA[2], boxB[2]);
            float yB = Mathf.Min(boxA[3], boxB[3]);
            
            // Python: interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            float interArea = Mathf.Max(0, xB - xA + 1) * Mathf.Max(0, yB - yA + 1);
            
            // Python: boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            // Python: boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            float boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
            float boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);
            
            // Python: iou = interArea / float(boxAArea + boxBArea - interArea)
            float iou = interArea / (boxAArea + boxBArea - interArea);
            
            return iou;
        }
        
        /// <summary>
        /// Invert 3x3 matrix - EXACT MATCH with numpy.linalg.inv
        /// </summary>
        private float[,] InvertMatrix3x3(float[,] matrix)
        {
            float[,] result = new float[3, 3];
            
            // Calculate determinant
            float det = matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
                      - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
                      + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]);
            
            if (Mathf.Abs(det) < 1e-6f)
            {
                throw new InvalidOperationException("Matrix is singular and cannot be inverted");
            }
            
            float invDet = 1.0f / det;
            
            // Calculate adjugate matrix and multiply by 1/det
            result[0, 0] = (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) * invDet;
            result[0, 1] = (matrix[0, 2] * matrix[2, 1] - matrix[0, 1] * matrix[2, 2]) * invDet;
            result[0, 2] = (matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1]) * invDet;
            
            result[1, 0] = (matrix[1, 2] * matrix[2, 0] - matrix[1, 0] * matrix[2, 2]) * invDet;
            result[1, 1] = (matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0]) * invDet;
            result[1, 2] = (matrix[0, 2] * matrix[1, 0] - matrix[0, 0] * matrix[1, 2]) * invDet;
            
            result[2, 0] = (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]) * invDet;
            result[2, 1] = (matrix[0, 1] * matrix[2, 0] - matrix[0, 0] * matrix[2, 1]) * invDet;
            result[2, 2] = (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]) * invDet;
            
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
        
        /// <summary>
        /// Python: x_d_new = scale_new * (x_c_s @ R_new + delta_new) + t_new - EXACT MATCH
        /// </summary>
        private float[] CalculateNewKeypoints(float[] xCs, float[,] RNew, float[] deltaNew, float[] scaleNew, float[] tNew)
        {
            int numKp = xCs.Length / 3;
            var result = new float[xCs.Length];
            
            for (int kp = 0; kp < numKp; kp++)
            {
                float x = xCs[kp * 3 + 0];
                float y = xCs[kp * 3 + 1]; 
                float z = xCs[kp * 3 + 2];
                
                // Matrix multiply: kp @ R_new
                float newX = x * RNew[0, 0] + y * RNew[1, 0] + z * RNew[2, 0];
                float newY = x * RNew[0, 1] + y * RNew[1, 1] + z * RNew[2, 1];
                float newZ = x * RNew[0, 2] + y * RNew[1, 2] + z * RNew[2, 2];
                
                // Add delta_new
                if (kp * 3 + 2 < deltaNew.Length)
                {
                    newX += deltaNew[kp * 3 + 0];
                    newY += deltaNew[kp * 3 + 1];
                    newZ += deltaNew[kp * 3 + 2];
                }
                
                // Multiply by scale_new
                if (scaleNew.Length > 0)
                {
                    newX *= scaleNew[0];
                    newY *= scaleNew[0];
                    newZ *= scaleNew[0];
                }
                
                // Add t_new
                if (tNew.Length >= 3)
                {
                    newX += tNew[0];
                    newY += tNew[1];
                    newZ += tNew[2];
                }
                
                result[kp * 3 + 0] = newX;
                result[kp * 3 + 1] = newY;
                result[kp * 3 + 2] = newZ;
            }
            
            return result;
        }
        
        /// <summary>
        /// Python: out.transpose(0, 2, 3, 1) and convert to texture - EXACT MATCH
        /// </summary>
        private Texture2D ConvertOutputToTexture(float[] output)
        {
            // CRITICAL FIX: Warping SPADE output is 1x3x512x512 as confirmed by logs!
            int channels = 3;
            int totalPixels = output.Length / channels;
            int size = Mathf.RoundToInt(Mathf.Sqrt(totalPixels));
            int height = size;
            int width = size;
            
            
            // Debug: Check output value ranges
            float minVal = output.Min();
            float maxVal = output.Max();
            
            var texture = new Texture2D(width, height, TextureFormat.RGB24, false);
            var pixels = new Color[width * height];
            
            // Python: out = out.transpose(0, 2, 3, 1)  # 1x3xHxW -> 1xHxWx3
            // Python: out = np.clip(out, 0, 1)  # clip to 0~1
            // Python: out = (out * 255).astype(np.uint8)  # 0~1 -> 0~255
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // CRITICAL: ONNX output is top-left origin, flip Y for Unity SetPixels (bottom-left)
                    int unityY = height - 1 - h; // Flip Y coordinate for Unity coordinate system
                    int pixelIdx = unityY * width + w;
                    
                    // CHW indexing
                    int rIdx = 0 * height * width + h * width + w;
                    int gIdx = 1 * height * width + h * width + w;
                    int bIdx = 2 * height * width + h * width + w;
                    
                    float r = Mathf.Clamp01(output[rIdx]);
                    float g = Mathf.Clamp01(output[gIdx]);
                    float b = Mathf.Clamp01(output[bIdx]);
                    
                    pixels[pixelIdx] = new Color(r, g, b, 1f);
                }
            }
            
            texture.SetPixels(pixels);
            texture.Apply();
            return texture;
        }
        
        // Face detection and landmark processing methods - SIMPLIFIED FOR NOW
        private DenseTensor<float> PreprocessDetectionImage(Texture2D img, int inputSize)
        {
            var pixels = img.GetPixels();
            var tensorData = new float[1 * 3 * inputSize * inputSize];
            
            int idx = 0;
            // Python: det_img = (det_img - 127.5) / 128
            // Python: det_img = det_img.transpose(2, 0, 1)  # HWC -> CHW
            // Python: det_img = np.expand_dims(det_img, axis=0)
            // Python: det_img = det_img.astype(np.float32)
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < inputSize; h++)
                {
                    for (int w = 0; w < inputSize; w++)
                    {
                        // CRITICAL: Unity GetPixels() is bottom-left origin, flip Y for ONNX (top-left)
                        int unityY = inputSize - 1 - h; // Flip Y coordinate for ONNX coordinate system
                        int pixelIdx = unityY * inputSize + w;
                        float pixelValue = c == 0 ? pixels[pixelIdx].r : 
                                          c == 1 ? pixels[pixelIdx].g : 
                                                   pixels[pixelIdx].b;
                        tensorData[idx++] = (pixelValue * 255f - 127.5f) / 128f;
                    }
                }
            }
            
            return new DenseTensor<float>(tensorData, new[] { 1, 3, inputSize, inputSize });
        }
        
        private List<FaceDetectionResult> ProcessDetectionResults(NamedOnnxValue[] outputs, float detScale)
        {
            
            // Python: process detection results exactly as in face_analysis function
            var scoresList = new List<float[]>();
            var bboxesList = new List<float[]>();
            var kpssList = new List<float[]>();
            
            const float detThresh = 0.5f;  // Python: det_thresh = 0.5
            const int fmc = 3;  // Python: fmc = 3
            int[] featStrideFpn = { 8, 16, 32 };  // Python: feat_stride_fpn = [8, 16, 32]
            const int inputSize = 512;
            var centerCache = new Dictionary<string, float[,]>();
            
            // Python: for idx, stride in enumerate(feat_stride_fpn):
            for (int idx = 0; idx < featStrideFpn.Length; idx++)
            {
                int stride = featStrideFpn[idx];
                
                // Python: scores = output[idx]
                var scoresTensor = outputs[idx].AsTensor<float>();
                // The scores tensor is often flattened by the ONNX runtime to [N, 1] or just [N].
                // We just need the flat array of scores, so we can call ToArray() directly.
                var scores = scoresTensor.ToArray();
                
                // Python: bbox_preds = output[idx + fmc]
                var bboxPredsTensor = outputs[idx + fmc].AsTensor<float>();
                var bboxPreds = bboxPredsTensor.ToArray();

                // Python: bbox_preds = bbox_preds * stride
                for (int i = 0; i < bboxPreds.Length; i++)
                {
                    bboxPreds[i] *= stride;
                }
                
                // Python: kps_preds = output[idx + fmc * 2] * stride
                var kpsPredsTensor = outputs[idx + fmc * 2].AsTensor<float>();
                var kpsPreds = kpsPredsTensor.ToArray();
                for (int i = 0; i < kpsPreds.Length; i++)
                {
                    kpsPreds[i] *= stride;
                }
                
                // Python: height = input_size // stride
                // Python: width = input_size // stride
                int height = inputSize / stride;
                int width = inputSize / stride;
                
                // Python: anchor_centers generation
                string key = $"{height}_{width}_{stride}";
                float[,] anchorCenters;
                
                if (centerCache.ContainsKey(key))
                {
                    anchorCenters = centerCache[key];
                }
                else
                {
                    // Python: anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    var centers = new List<Vector2>();
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            centers.Add(new Vector2(w, h));  // [::-1] means reverse order
                        }
                    }
                    
                    // Python: anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                    // Python: anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
                    const int numAnchors = 2;
                    anchorCenters = new float[centers.Count * numAnchors, 2];
                    
                    for (int i = 0; i < centers.Count; i++)
                    {
                        for (int a = 0; a < numAnchors; a++)
                        {
                            int idx2 = i * numAnchors + a;
                            anchorCenters[idx2, 0] = centers[i].x * stride;
                            anchorCenters[idx2, 1] = centers[i].y * stride;
                        }
                    }
                    
                    if (centerCache.Count < 100)
                    {
                        centerCache[key] = anchorCenters;
                    }
                }
                
                // Python: pos_inds = np.where(scores >= det_thresh)[0]
                var posInds = new List<int>();
                for (int i = 0; i < scores.Length; i++)
                {
                    if (scores[i] >= detThresh)
                    {
                        posInds.Add(i);
                    }
                }
                
                // if (scores.Length > 0)
                // {
                // }
                
                if (posInds.Count > 0)
                {
                    // Python: bboxes = distance2bbox(anchor_centers, bbox_preds)
                    var bboxes = Distance2Bbox(anchorCenters, bboxPreds);
                    
                    // Python: pos_scores = scores[pos_inds]
                    var posScores = new float[posInds.Count];
                    for (int i = 0; i < posInds.Count; i++)
                    {
                        posScores[i] = scores[posInds[i]];
                    }
                    
                    // Python: pos_bboxes = bboxes[pos_inds]
                    var posBboxes = new float[posInds.Count * 4];
                    for (int i = 0; i < posInds.Count; i++)
                    {
                        int srcIdx = posInds[i];
                        posBboxes[i * 4 + 0] = bboxes[srcIdx * 4 + 0];
                        posBboxes[i * 4 + 1] = bboxes[srcIdx * 4 + 1];
                        posBboxes[i * 4 + 2] = bboxes[srcIdx * 4 + 2];
                        posBboxes[i * 4 + 3] = bboxes[srcIdx * 4 + 3];
                    }
                    
                    scoresList.Add(posScores);
                    bboxesList.Add(posBboxes);
                    
                    // Python: kpss = distance2kps(anchor_centers, kps_preds)
                    var kpss = Distance2Kps(anchorCenters, kpsPreds);
                    
                    // Python: pos_kpss = kpss[pos_inds]
                    var posKpss = new float[posInds.Count * 10]; // 5 keypoints * 2 coords
                    for (int i = 0; i < posInds.Count; i++)
                    {
                        int srcIdx = posInds[i];
                        for (int k = 0; k < 10; k++)
                        {
                            posKpss[i * 10 + k] = kpss[srcIdx * 10 + k];
                        }
                    }
                    
                    kpssList.Add(posKpss);
                }
            }
            
            if (scoresList.Count == 0)
            {
                return new List<FaceDetectionResult>();
            }
            
            // Python: scores = np.vstack(scores_list)
            var allScores = new List<float>();
            var allBboxes = new List<float>();
            var allKpss = new List<float>();
            
            foreach (var scores in scoresList)
            {
                allScores.AddRange(scores);
            }
            
            foreach (var bboxes in bboxesList)
            {
                allBboxes.AddRange(bboxes);
            }
            
            foreach (var kpss in kpssList)
            {
                allKpss.AddRange(kpss);
            }
            
            // Python: scores_ravel = scores.ravel()
            // Python: order = scores_ravel.argsort()[::-1]
            var scoreIndices = new List<(float score, int index)>();
            for (int i = 0; i < allScores.Count; i++)
            {
                scoreIndices.Add((allScores[i], i));
            }
            scoreIndices.Sort((a, b) => b.score.CompareTo(a.score)); // Descending order
            
            // CRITICAL: Scale bboxes and keypoints by detScale to convert back to original image coordinates
            // Python: bboxes /= det_scale, kpss /= det_scale
            for (int i = 0; i < allBboxes.Count; i++)
            {
                allBboxes[i] /= detScale;
            }
            
            for (int i = 0; i < allKpss.Count; i++)
            {
                allKpss[i] /= detScale;
            }
            
            // Python: pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
            // Python: pre_det = pre_det[order, :]
            // Python: kpss = kpss[order, :, :]
            var preDet = new List<float[]>();
            var kpssSorted = new List<float>();
            foreach (var item in scoreIndices)
            {
                int originalIndex = item.index;

                var det = new float[5];
                det[0] = allBboxes[originalIndex * 4 + 0];
                det[1] = allBboxes[originalIndex * 4 + 1];
                det[2] = allBboxes[originalIndex * 4 + 2];
                det[3] = allBboxes[originalIndex * 4 + 3];
                det[4] = item.score;
                preDet.Add(det);

                // Add corresponding keypoints
                for (int k = 0; k < 10; k++)
                {
                    kpssSorted.Add(allKpss[originalIndex * 10 + k]);
                }
            }
            
            // Python: keep = nms_boxes(pre_det, [1 for s in pre_det], nms_thresh)
            const float nmsThresh = 0.4f;
            var scoresForNms = Enumerable.Repeat(1f, preDet.Count).ToList();
            var keep = NmsBoxes(preDet, scoresForNms, nmsThresh);
            
            // Build final face detection results
            var faces = new List<FaceDetectionResult>();
            
            foreach (int keepIdx in keep)
            {
                var bbox = preDet[keepIdx];
                
                // CRITICAL: Store bounding box in OpenCV coordinates (top-left origin) as Python expects
                // Python bbox format: [x1, y1, x2, y2] in original image coordinates
                var face = new FaceDetectionResult
                {
                    BoundingBox = new Rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                    DetectionScore = bbox[4],
                    Keypoints5 = new Vector2[5],
                    Landmarks106 = new Vector2[106] // Will be filled later
                };
                
                // Store keypoints in original image coordinates (OpenCV format)
                for (int k = 0; k < 5; k++)
                {
                    face.Keypoints5[k] = new Vector2(
                        kpssSorted[keepIdx * 10 + k * 2],
                        kpssSorted[keepIdx * 10 + k * 2 + 1]
                    );
                }
                
                faces.Add(face);
            }
            
            return faces;
        }
        
        /// <summary>
        /// Python: face_align(data, center, output_size, scale, rotation) - EXACT MATCH
        /// </summary>
        private (Texture2D, Matrix4x4) FaceAlign(Texture2D img, Vector2 center, int inputSize, float scale, float rotate)
        {
            // Python: scale_ratio = scale
            float scaleRatio = scale;
            // Python: rot = float(rotation) * np.pi / 180.0
            float rot = rotate * Mathf.Deg2Rad;
            
            // Python: trans_M = np.array([[scale_ratio*cos(rot), -scale_ratio*sin(rot), output_size*0.5-center[0]*scale_ratio*cos(rot) + center[1]*scale_ratio*sin(rot)], 
            //                            [scale_ratio*sin(rot), scale_ratio*cos(rot), output_size*0.5-center[0]*scale_ratio*sin(rot) - center[1]*scale_ratio*cos(rot)], 
            //                            [0, 0, 1]], dtype=np.float32)
            // Simplified from python:
            // t_x = output_size*0.5 - (center[0]*scale_ratio*cos(rot) - center[1]*scale_ratio*sin(rot))
            // t_y = output_size*0.5 - (center[0]*scale_ratio*sin(rot) + center[1]*scale_ratio*cos(rot))
            float cosRot = Mathf.Cos(rot);
            float sinRot = Mathf.Sin(rot);
            float outputSizeHalf = inputSize * 0.5f;
            
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
            
            // Python: cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
            var cropped = TransformImgExact(img, M, inputSize);

            // Convert to Matrix4x4 for Unity compatibility
            var transform = new Matrix4x4
            {
                m00 = M[0, 0],
                m01 = M[0, 1],
                m02 = 0f,
                m03 = M[0, 2],
                m10 = M[1, 0],
                m11 = M[1, 1],
                m12 = 0f,
                m13 = M[1, 2],
                m20 = 0f,
                m21 = 0f,
                m22 = 1f,
                m23 = 0f,
                m30 = 0f,
                m31 = 0f,
                m32 = 0f,
                m33 = 1f
            };

            return (cropped, transform);
        }
        
        private DenseTensor<float> PreprocessLandmarkImage(Texture2D img, int inputSize = 192)
        {
            var pixels = img.GetPixels32();
            var tensorData = new float[1 * 3 * inputSize * inputSize];
            
            int idx = 0;
            // The following loops perform the equivalent of numpy's transpose(2, 0, 1)
            // to convert from HWC (height, width, channel) to CHW (channel, height, width).
            for (int c = 0; c < 3; c++) // Channel
            {
                for (int h = 0; h < inputSize; h++) // Height
                {
                    for (int w = 0; w < inputSize; w++) // Width
                    {
                        // CRITICAL: Unity GetPixels() is bottom-left origin, flip Y for ONNX (top-left)
                        int unityY = inputSize - 1 - h; // Flip Y coordinate for ONNX coordinate system
                        int pixelIdx = unityY * inputSize + w;
                        float pixelValue = c == 0 ? pixels[pixelIdx].r : 
                                          c == 1 ? pixels[pixelIdx].g : 
                                                   pixels[pixelIdx].b;
                        // CRITICAL FIX: Python does NOT normalize to [0,1] for landmark detection!
                        // Keep pixel values in [0,255] range to match Python exactly
                        tensorData[idx++] = pixelValue; // Convert from [0,1] to [0,255]
                    }
                }
            }
            
            // The DenseTensor is created with a shape that includes the batch dimension (1),
            // which is equivalent to numpy's expand_dims(axis=0).
            return new DenseTensor<float>(tensorData, new[] { 1, 3, inputSize, inputSize });
        }
        
        private Matrix4x4 InvertAffineTransform(Matrix4x4 matrix)
        {
            return matrix.inverse;
        }
        
        /// <summary>
        /// Invert Matrix4x4 using the same method as the 2x3 affine transform for consistency
        /// </summary>
        private float[,] InvertAffineTransformToMatrix(Matrix4x4 matrix)
        {
            // Extract 2x3 transformation matrix
            float[,] M = new float[2, 3] {
                { matrix.m00, matrix.m01, matrix.m03 },
                { matrix.m10, matrix.m11, matrix.m13 }
            };
            
            return InvertAffineTransform(M);
        }
        
        /// <summary>
        /// Python: cv2.invertAffineTransform(M) - EXACT MATCH
        /// Invert a 2x3 affine transformation matrix
        /// </summary>
        private float[,] InvertAffineTransform(float[,] M)
        {
            // For 2x3 matrix [[a, b, c], [d, e, f]], the inverse is:
            // det = a*e - b*d
            // inv = [[e/det, -b/det, (b*f-c*e)/det], [-d/det, a/det, (c*d-a*f)/det]]
            
            float a = M[0, 0], b = M[0, 1], c = M[0, 2];
            float d = M[1, 0], e = M[1, 1], f = M[1, 2];
            
            float det = a * e - b * d;
            
            if (Mathf.Abs(det) < 1e-6f)
            {
                throw new InvalidOperationException("Affine matrix is singular and cannot be inverted");
            }
            
            float[,] inv = new float[2, 3];
            inv[0, 0] = e / det;
            inv[0, 1] = -b / det;
            inv[0, 2] = (b * f - c * e) / det;
            inv[1, 0] = -d / det;
            inv[1, 1] = a / det;
            inv[1, 2] = (c * d - a * f) / det;
            
            return inv;
        }
        

        
        private Vector2[] TransformPoints2D(Vector2[] points, Matrix4x4 transform)
        {
            var result = new Vector2[points.Length];
            for (int i = 0; i < points.Length; i++)
            {
                var transformed = transform.MultiplyPoint3x4(new Vector3(points[i].x, points[i].y, 0));
                result[i] = new Vector2(transformed.x, transformed.y);
            }
            return result;
        }
        
        private DenseTensor<float> PreprocessLandmarkRunnerImage(Texture2D img)
        {
            var pixels = img.GetPixels();
            var tensorData = new float[1 * 3 * 224 * 224];
            
            int idx = 0;
            // Python: img_crop = img_crop / 255
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 224; h++)
                {
                    for (int w = 0; w < 224; w++)
                    {
                        // CRITICAL: Unity GetPixels() is bottom-left origin, flip Y for ONNX (top-left)
                        int unityY = 224 - 1 - h; // Flip Y coordinate for ONNX coordinate system
                        int pixelIdx = unityY * 224 + w;
                        float pixelValue = c == 0 ? pixels[pixelIdx].r : 
                                          c == 1 ? pixels[pixelIdx].g : 
                                                   pixels[pixelIdx].b;
                        tensorData[idx++] = pixelValue; // Already normalized [0,1]
                    }
                }
            }
            
            return new DenseTensor<float>(tensorData, new[] { 1, 3, 224, 224 });
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
        /// Python: trans_points2d() - EXACT MATCH
        /// </summary>
        private Vector2[] TransPoints2D(Vector2[] pts, Matrix4x4 M)
        {
            var result = new Vector2[pts.Length];
            
            for (int i = 0; i < pts.Length; i++)
            {
                Vector2 pt = pts[i];
                // Python: new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
                // Python: new_pt = np.dot(M, new_pt)
                // Python: new_pts[i] = new_pt[0:2]
                Vector3 newPt = new Vector3(pt.x, pt.y, 1.0f);
                Vector3 transformed = M.MultiplyPoint3x4(newPt);
                result[i] = new Vector2(transformed.x, transformed.y);
            }
            
            return result;
        }
        
        /// <summary>
        /// Python: trans_points2d() with 2x3 matrix - EXACT MATCH
        /// </summary>
        private Vector2[] TransPoints2D(Vector2[] pts, float[,] M)
        {
            var result = new Vector2[pts.Length];
            
            for (int i = 0; i < pts.Length; i++)
            {
                Vector2 pt = pts[i];
                // Python: new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
                // Python: new_pt = np.dot(M, new_pt)
                result[i] = new Vector2(
                    M[0, 0] * pt.x + M[0, 1] * pt.y + M[0, 2],
                    M[1, 0] * pt.x + M[1, 1] * pt.y + M[1, 2]
                );
            }
            
            return result;
        }
        
        /// <summary>
        /// Python: cv2.warpAffine() - EXACT MATCH
        /// This is the corrected version that matches OpenCV's warpAffine exactly
        /// CRITICAL: Handles coordinate systems correctly
        /// </summary>
        private Texture2D TransformImgExact(Texture2D img, float[,] M, int dsize)
        {
            // Create result texture - MUST use RGB24 format for consistent processing
            var result = new Texture2D(dsize, dsize, TextureFormat.RGB24, false);

            // Get source image data as raw pixel array - Unity format is RGBA but we need RGB
            var srcPixels = img.GetPixels32(); // Get as Color32 for better precision
            int srcWidth = img.width;
            int srcHeight = img.height;
            
            // Create result pixel array
            var resultPixels = new Color32[dsize * dsize];

            // Invert the transformation matrix M to get the mapping from destination to source
            float[,] invM = InvertAffineTransform(M);

            // CRITICAL: Process exactly like OpenCV cv2.warpAffine
            // OpenCV processes in row-major order with top-left origin (0,0) at top-left
            for (int dstY = 0; dstY < dsize; dstY++)
            {
                for (int dstX = 0; dstX < dsize; dstX++)
                {
                    // Apply inverse transformation matrix to find source coordinates
                    float srcX = invM[0, 0] * dstX + invM[0, 1] * dstY + invM[0, 2];
                    float srcY = invM[1, 0] * dstX + invM[1, 1] * dstY + invM[1, 2];

                    // Get integer and fractional parts for bilinear interpolation
                    int x0 = Mathf.FloorToInt(srcX);
                    int y0 = Mathf.FloorToInt(srcY);

                    float fx = srcX - x0;
                    float fy = srcY - y0;

                    // Default to black (borderValue=0.0 in OpenCV)
                    byte r = 0, g = 0, b = 0;

                    // Bounds check for bilinear interpolation
                    if (x0 >= 0 && (x0 + 1) < srcWidth && y0 >= 0 && (y0 + 1) < srcHeight)
                    {
                        // CRITICAL: OpenCV uses top-left origin, Unity GetPixels32() uses bottom-left origin
                        // Convert OpenCV coordinates to Unity coordinates for pixel access
                        int unity_y0 = srcHeight - 1 - y0;
                        int unity_y1 = srcHeight - 1 - (y0 + 1);

                        // Get the four corner pixels for bilinear interpolation
                        var c00 = srcPixels[unity_y0 * srcWidth + x0];
                        var c10 = srcPixels[unity_y0 * srcWidth + x0 + 1];
                        var c01 = srcPixels[unity_y1 * srcWidth + x0];
                        var c11 = srcPixels[unity_y1 * srcWidth + x0 + 1];

                        // Bilinear interpolation
                        float inv_fx = 1.0f - fx;
                        float inv_fy = 1.0f - fy;
                        
                        float r_float = inv_fx * inv_fy * c00.r + fx * inv_fy * c10.r + inv_fx * fy * c01.r + fx * fy * c11.r;
                        float g_float = inv_fx * inv_fy * c00.g + fx * inv_fy * c10.g + inv_fx * fy * c01.g + fx * fy * c11.g;
                        float b_float = inv_fx * inv_fy * c00.b + fx * inv_fy * c10.b + inv_fx * fy * c01.b + fx * fy * c11.b;

                        r = (byte)Mathf.Clamp(r_float, 0f, 255f);
                        g = (byte)Mathf.Clamp(g_float, 0f, 255f);
                        b = (byte)Mathf.Clamp(b_float, 0f, 255f);
                    }

                    // Store result pixel.
                    // Unity's SetPixels32 expects a 1D array that's row-major, starting from bottom-left.
                    // Our outer loop (dstY) iterates from top to bottom, so we write to the array accordingly.
                    int result_idx = (dsize - 1 - dstY) * dsize + dstX;
                    resultPixels[result_idx] = new Color32(r, g, b, 255);
                }
            }

            result.SetPixels32(resultPixels);
            result.Apply();
            return result;
        }
        
        public void Dispose()
        {
            if (_disposed) return;
            
            try
            {
                _detFace?.Dispose();
                _landmark2d106?.Dispose();
                _landmarkRunner?.Dispose();
                _appearanceFeatureExtractor?.Dispose();
                _motionExtractor?.Dispose();
                _stitching?.Dispose();
                _warpingSpade?.Dispose();
                _insightFaceHelper?.Dispose();
                
                // Clean up mask template if it was loaded during initialization
                if (_maskTemplate != null)
                {
                    _maskTemplate = null;
                }
                
                _disposed = true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[LivePortraitInference] Error during disposal: {e.Message}");
            }
        }
        
        /// <summary>
        /// Python: prepare_paste_back(mask_crop, crop_M_c2o, dsize) - EXACT MATCH
        /// </summary>
        private Texture2D PreparePasteBack(Matrix4x4 cropMc2o, int width, int height)
        {
            // Python: mask_ori = cv2.warpAffine(mask_crop, crop_M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR)
            // Python: mask_ori = mask_ori.astype(np.float32) / 255.0
            
            if (_maskTemplate == null)
            {
                Debug.LogWarning("[PreparePasteBack] No mask template provided, creating default circular mask");
                return CreateDefaultMask(width, height);
            }
            
            // Transform mask template using crop transformation matrix
            float[,] M = new float[,] {
                { cropMc2o.m00, cropMc2o.m01, cropMc2o.m03 },
                { cropMc2o.m10, cropMc2o.m11, cropMc2o.m13 }
            };
            
            var maskOri = TransformImgExact(_maskTemplate, M, width, height);

            // In Unity, GetPixels() on an RGB24 texture returns float Colors in the [0,1] range, matching Python's / 255.0.
            // The Python script does not convert to grayscale, so we remove the manual pixel loop.
            // The warped 3-channel mask is returned directly.
            
            
            return maskOri;
        }
        
        /// <summary>
        /// Create default circular mask when no template is provided
        /// </summary>
        private Texture2D CreateDefaultMask(int width, int height)
        {
            var maskOri = new Texture2D(width, height, TextureFormat.RGB24, false);
            var pixels = new Color[width * height];
            
            // Create a circular/elliptical mask in the center region
            Vector2 center = new Vector2(width * 0.5f, height * 0.5f);
            float radiusX = width * 0.3f;  // Elliptical mask
            float radiusY = height * 0.4f;
            
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int pixelIdx = y * width + x;
                    
                    // Calculate distance from center (elliptical)
                    float dx = (x - center.x) / radiusX;
                    float dy = (y - center.y) / radiusY;
                    float distance = Mathf.Sqrt(dx * dx + dy * dy);
                    
                    // Smooth falloff from center
                    float maskValue;
                    if (distance <= 0.8f)
                    {
                        maskValue = 1.0f; // Full mask in center
                    }
                    else if (distance <= 1.2f)
                    {
                        // Smooth transition
                        maskValue = Mathf.Lerp(1.0f, 0.0f, (distance - 0.8f) / 0.4f);
                    }
                    else
                    {
                        maskValue = 0.0f; // No mask at edges
                    }
                    
                    pixels[pixelIdx] = new Color(maskValue, maskValue, maskValue, 1f);
                }
            }
            
            maskOri.SetPixels(pixels);
            maskOri.Apply();
            
            
            return maskOri;
        }
        
        /// <summary>
        /// Python: concat_frame(img_rgb, img_crop_256x256, I_p) - EXACT MATCH
        /// </summary>
        private Texture2D ConcatFrame(Texture2D imgRgb, Texture2D imgCrop256x256, Texture2D Ip)
        {
            // Python: Concatenate frames horizontally: driving | cropped | generated
            int width = imgRgb.width + imgCrop256x256.width + Ip.width;
            int height = Mathf.Max(imgRgb.height, Mathf.Max(imgCrop256x256.height, Ip.height));
            
            var result = new Texture2D(width, height, TextureFormat.RGB24, false);
            var pixels = new Color[width * height];
            
            // Fill with black background
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = Color.black;
            }
            
            result.SetPixels(pixels);
            
            // Copy driving image
            var drivingPixels = imgRgb.GetPixels();
            result.SetPixels(0, 0, imgRgb.width, imgRgb.height, drivingPixels);
            
            // Copy cropped image
            var croppedPixels = imgCrop256x256.GetPixels();
            result.SetPixels(imgRgb.width, 0, imgCrop256x256.width, imgCrop256x256.height, croppedPixels);
            
            // Copy generated image
            var generatedPixels = Ip.GetPixels();
            result.SetPixels(imgRgb.width + imgCrop256x256.width, 0, Ip.width, Ip.height, generatedPixels);
            
            result.Apply();
            return result;
        }
        
        /// <summary>
        /// Python: paste_back(img_crop, M_c2o, img_ori, mask_ori) - EXACT MATCH
        /// </summary>
        private Texture2D PasteBack(Texture2D imgCrop, Matrix4x4 Mc2o, Texture2D imgOri, Texture2D maskOri)
        {
            
            // Python: dsize = (img_ori.shape[1], img_ori.shape[0])
            int dsize_w = imgOri.width;
            int dsize_h = imgOri.height;
            
            // Debug original image pixel values
            var oriPixelsDebug = imgOri.GetPixels();
            float oriMin = oriPixelsDebug.Min(p => Mathf.Min(p.r, Mathf.Min(p.g, p.b)));
            float oriMax = oriPixelsDebug.Max(p => Mathf.Max(p.r, Mathf.Max(p.g, p.b)));
            
            // Python: result = cv2.warpAffine(img_crop, M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR)
            float[,] M = new float[,] {
                { Mc2o.m00, Mc2o.m01, Mc2o.m03 },
                { Mc2o.m10, Mc2o.m11, Mc2o.m13 }
            };
            var warped = TransformImgExact(imgCrop, M, dsize_w, dsize_h);
            
            // Debug warped image pixel values
            var warpedPixelsDebug = warped.GetPixels();
            float warpedMin = warpedPixelsDebug.Min(p => Mathf.Min(p.r, Mathf.Min(p.g, p.b)));
            float warpedMax = warpedPixelsDebug.Max(p => Mathf.Max(p.r, Mathf.Max(p.g, p.b)));
            
            // Python: result = np.clip(mask_ori * result + (1 - mask_ori) * img_ori, 0, 255).astype(np.uint8)
            var result = new Texture2D(dsize_w, dsize_h, TextureFormat.RGB24, false);
            var warpedPixels = warped.GetPixels();
            var oriPixels = imgOri.GetPixels();
            var maskPixels = maskOri.GetPixels();
            var resultPixels = new Color[dsize_w * dsize_h];
            
            // Check array sizes match
            if (warpedPixels.Length != oriPixels.Length || oriPixels.Length != maskPixels.Length)
            {
                Debug.LogError($"[DEBUG_PASTEBACK] Pixel array size mismatch! warped: {warpedPixels.Length}, ori: {oriPixels.Length}, mask: {maskPixels.Length}");
            }
            
            for (int i = 0; i < resultPixels.Length && i < warpedPixels.Length && i < oriPixels.Length && i < maskPixels.Length; i++)
            {
                // Python: result = np.clip(mask_ori * result + (1 - mask_ori) * img_ori, 0, 255)
                // This is a per-channel blend, equivalent to Lerp(ori, warped, mask) for each channel.
                Color warpedP = warpedPixels[i];
                Color oriP = oriPixels[i];
                Color maskP = maskPixels[i];
                
                float r = oriP.r * (1f - maskP.r) + warpedP.r * maskP.r;
                float g = oriP.g * (1f - maskP.g) + warpedP.g * maskP.g;
                float b = oriP.b * (1f - maskP.b) + warpedP.b * maskP.b;
                
                resultPixels[i] = new Color(r, g, b, 1f);
            }
            
            result.SetPixels(resultPixels);
            result.Apply();
            
            // Debug final result pixel values
            var resultPixelsDebug = result.GetPixels();
            float resultMin = resultPixelsDebug.Min(p => Mathf.Min(p.r, Mathf.Min(p.g, p.b)));
            float resultMax = resultPixelsDebug.Max(p => Mathf.Max(p.r, Mathf.Max(p.g, p.b)));
            
            // UnityEngine.Object.DestroyImmediate(warped);
            
            return result;
        }
        
        /// <summary>
        /// Overload for TransformImgExact with different dimensions
        /// </summary>
        private Texture2D TransformImgExact(Texture2D img, float[,] M, int width, int height)
        {
            // Create result texture - MUST use RGB24 format for consistent processing
            var result = new Texture2D(width, height, TextureFormat.RGB24, false);
            
            // Get source image data as raw pixel array
            var srcPixels = img.GetPixels32();
            int srcWidth = img.width;
            int srcHeight = img.height;
            
            // Create result pixel array
            var resultPixels = new Color32[width * height];
            
            // Invert the transformation matrix M to get the mapping from destination to source
            float[,] invM = InvertAffineTransform(M);
            
            // Process exactly like OpenCV cv2.warpAffine
            for (int dstY = 0; dstY < height; dstY++)
            {
                for (int dstX = 0; dstX < width; dstX++)
                {
                    // Apply inverse transformation matrix
                    float srcX = invM[0, 0] * dstX + invM[0, 1] * dstY + invM[0, 2];
                    float srcY = invM[1, 0] * dstX + invM[1, 1] * dstY + invM[1, 2];
                    
                    // Get integer and fractional parts for bilinear interpolation
                    int x0 = Mathf.FloorToInt(srcX);
                    int y0 = Mathf.FloorToInt(srcY);

                    float fx = srcX - x0;
                    float fy = srcY - y0;
                    
                    // Default to black
                    byte r = 0, g = 0, b = 0;
                    
                    // Bounds check for bilinear interpolation
                    if (x0 >= 0 && (x0 + 1) < srcWidth && y0 >= 0 && (y0 + 1) < srcHeight)
                    {
                        // Convert OpenCV coordinates to Unity coordinates
                        int unity_y0 = srcHeight - 1 - y0;
                        int unity_y1 = srcHeight - 1 - (y0 + 1);
                        
                        // Get the four corner pixels
                        var c00 = srcPixels[unity_y0 * srcWidth + x0];
                        var c10 = srcPixels[unity_y0 * srcWidth + x0 + 1];
                        var c01 = srcPixels[unity_y1 * srcWidth + x0];
                        var c11 = srcPixels[unity_y1 * srcWidth + x0 + 1];
                        
                        // Bilinear interpolation
                        float inv_fx = 1.0f - fx;
                        float inv_fy = 1.0f - fy;
                        
                        float r_float = inv_fx * inv_fy * c00.r + fx * inv_fy * c10.r + inv_fx * fy * c01.r + fx * fy * c11.r;
                        float g_float = inv_fx * inv_fy * c00.g + fx * inv_fy * c10.g + inv_fx * fy * c01.g + fx * fy * c11.g;
                        float b_float = inv_fx * inv_fy * c00.b + fx * inv_fy * c10.b + inv_fx * fy * c01.b + fx * fy * c11.b;
                        
                        r = (byte)Mathf.Clamp(r_float, 0f, 255f);
                        g = (byte)Mathf.Clamp(g_float, 0f, 255f);
                        b = (byte)Mathf.Clamp(b_float, 0f, 255f);
                    }
                    
                    // Store result pixel
                    int result_idx = (height - 1 - dstY) * width + dstX;
                    resultPixels[result_idx] = new Color32(r, g, b, 255);
                }
            }
            
            result.SetPixels32(resultPixels);
            result.Apply();
            return result;
        }
        
        /// <summary>
        /// Python: calculate_distance_ratio(lmk, idx1, idx2, idx3, idx4, eps=1e-6) - EXACT MATCH
        /// Calculate the ratio between two distances
        /// CRITICAL: Python expects lmk with batch dimension: (batch_size, num_landmarks, 2)
        /// </summary>
        private float[] CalculateDistanceRatio(Vector2[] lmk, int idx1, int idx2, int idx3, int idx4, float eps = 1e-6f)
        {
            // CRITICAL FIX: Python function expects batched landmarks lmk[:, idx1] means lmk[batch_idx, landmark_idx]
            // Since we have batch_size=1, lmk[:, idx1] becomes lmk[0, idx1] which is just lmk[idx1]
            // Python: d1 = np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True)
            // Python: d2 = np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True)
            // Python: ratio = d1 / (d2 + eps)
            
            // For batch_size=1: lmk[:, idx1] = lmk[0, idx1] = lmk[idx1]
            Vector2 p1 = lmk[idx1];
            Vector2 p2 = lmk[idx2];
            Vector2 p3 = lmk[idx3];
            Vector2 p4 = lmk[idx4];
            
            // np.linalg.norm(p1 - p2, axis=1, keepdims=True) with axis=1 means norm across coordinate dimension
            // For 2D points, this is just the Euclidean distance
            float d1 = Vector2.Distance(p1, p2);
            float d2 = Vector2.Distance(p3, p4);
            
            float ratio = d1 / (d2 + eps);
            
            // Return as array to match Python's keepdims=True behavior (shape becomes (1,))
            return new float[] { ratio };
        }
        
        /// <summary>
        /// Ensure array is float32 precision - matches Python's .astype(np.float32)
        /// </summary>
        private float[] EnsureFloat32Array(float[] array)
        {
            if (array == null) return null;
            
            // Create new array to ensure float32 precision (C# float is already float32, but this ensures a copy)
            var result = new float[array.Length];
            Array.Copy(array, result, array.Length);
            return result;
        }
        
        /// <summary>
        /// Ensure matrix is float32 precision - matches Python's .astype(np.float32)
        /// </summary>
        private float[,] EnsureFloat32Matrix(float[,] matrix)
        {
            if (matrix == null) return null;
            
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j];
                }
            }
            
            return result;
        }
    }

    /// <summary>
    /// Motion information extracted from face keypoints - matches Python kp_info structure
    /// </summary>
    public class MotionInfo
    {
        public float[] Pitch { get; set; }       // Processed pitch angles
        public float[] Yaw { get; set; }         // Processed yaw angles  
        public float[] Roll { get; set; }        // Processed roll angles
        public float[] Translation { get; set; } // t: translation parameters
        public float[] Expression { get; set; }  // exp: expression deformation
        public float[] Scale { get; set; }       // scale: scaling factor
        public float[] Keypoints { get; set; }   // kp: 3D keypoints
        public float[,] RotationMatrix { get; set; } // R_d: rotation matrix (added for Python compatibility)
    }
}

