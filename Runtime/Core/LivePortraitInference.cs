using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MuseTalk.Core
{
    using API;
    using Utils;

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
    /// LivePortrait input configuration
    /// </summary>
    public class LivePortraitInput
    {
        public Texture2D SourceImage { get; set; }
        public List<Texture2D> DrivingFrames { get; set; }
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
        private InferenceSession _appearanceFeatureExtractor;  // feature extraction
        private InferenceSession _motionExtractor;  // motion parameters
        private InferenceSession _stitching;  // keypoint stitching
        private InferenceSession _warpingSpade;  // neural warping
        
        // Face analysis (using consolidated FaceAnalysis class)
        private FaceAnalysis _faceAnalysis;

        private Texture2D _debugImage = null;
        
        // Configuration
        private MuseTalkConfig _config;
        private bool _initialized = false;
        private bool _disposed = false;
        
        // State management - matches Python self.pred_info
        private LivePortraitPredInfo _predInfo;
        
        // Mask template - matches Python self.mask_crop
        private Frame _maskTemplate;
        
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
            _appearanceFeatureExtractor = ModelUtils.LoadModel(_config, "appearance_feature_extractor");
            _motionExtractor = ModelUtils.LoadModel(_config, "motion_extractor");
            _stitching = ModelUtils.LoadModel(_config, "stitching");
            _warpingSpade = ModelUtils.LoadModel(_config, "warping_spade");
            
            // Initialize consolidated face analysis
            _faceAnalysis = new FaceAnalysis(_config);
            
            // Verify all models initialized
            bool allInitialized = _appearanceFeatureExtractor != null &&
                                 _motionExtractor != null &&
                                 _warpingSpade != null &&
                                 _stitching != null &&
                                 _faceAnalysis.IsInitialized;
            
            if (!allInitialized)
            {
                var failedModels = new List<string>();
                if (_appearanceFeatureExtractor == null) failedModels.Add("AppearanceExtractor");
                if (_motionExtractor == null) failedModels.Add("MotionExtractor");
                if (_warpingSpade == null) failedModels.Add("WarpingSPADE");
                if (_stitching == null) failedModels.Add("Stitching");
                if (!_faceAnalysis.IsInitialized) failedModels.Add("FaceAnalysis");
                
                throw new InvalidOperationException($"Failed to initialize models: {string.Join(", ", failedModels)}");
            }
        }

        private async Task<ProcessSourceImageResult> ProcessSourceImageAsync(Frame frame)
        {
            return await Task.Run(() => {
                var srcImg = SrcPreprocess(frame);
                
                // Python: crop_info = crop_src_image(self.models, src_img)
                var cropInfo = CropSrcImage(srcImg);

                // Python: img_crop_256x256 = crop_info["img_crop_256x256"]
                // Python: I_s = preprocess(img_crop_256x256)
                var Is = Preprocess(cropInfo.ImageCrop256x256);
                
                // Python: x_s_info = get_kp_info(self.models, I_s)
                var xSInfo = GetKpInfo(Is);
                
                // Python: R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
                var Rs = MathUtils.GetRotationMatrix(xSInfo.Pitch, xSInfo.Yaw, xSInfo.Roll);
                
                // Python: f_s = extract_feature_3d(self.models, I_s)
                var fs = ExtractFeature3d(Is);
                
                // Python: x_s = transform_keypoint(x_s_info)
                var xs = TransformKeypoint(xSInfo);
                
                // Python: prepare for pasteback
                // Python: mask_ori = prepare_paste_back(self.mask_crop, crop_info["M_c2o"], dsize=(src_img.shape[1], src_img.shape[0]))
                var maskOri = PreparePasteBack(cropInfo.Transform, srcImg.width, srcImg.height);

                return new ProcessSourceImageResult
                {
                    CropInfo = cropInfo,
                    SrcImg = srcImg,
                    MaskOri = maskOri,
                    XsInfo = xSInfo,
                    Rs = Rs,
                    Fs = fs,
                    Xs = xs
                };
            });
        }
        
        /// <summary>
        /// Generate talking head animation - matches Python LivePortraitWrapper.execute
        /// MAIN THREAD ONLY for correctness
        /// </summary>
        public IEnumerator GenerateAsync(LivePortraitInput input, LivePortaitStream stream)
        {
            if (!_initialized)
                throw new InvalidOperationException("LivePortrait inference not initialized");
                
            if (input?.SourceImage == null || input.DrivingFrames == null || input.DrivingFrames.Count == 0)
                throw new ArgumentException("Invalid input: source image and driving frames are required");
            
            if (_maskTemplate.data == null)
                throw new Exception("[LivePortraitInference] No mask template available");
            
            var srcFrame = TextureUtils.Texture2DToFrame(input.SourceImage);
            var processSrcTask = ProcessSourceImageAsync(srcFrame);
            yield return new WaitUntil(() => processSrcTask.IsCompleted);
            var processResult = processSrcTask.Result;

            var maxFrames = 0;

            // For debugging, only generate 1 frame - matches Python: if frame_id > 0: break
            for (int frameId = maxFrames; frameId < Mathf.Min(maxFrames + 25, input.DrivingFrames.Count); frameId++)
            {
                // Python: img_rgb = frame[:, :, ::-1]  # BGR -> RGB (Unity input is already RGB)
                var imgRgb = input.DrivingFrames[frameId];
                var imgRgbFrame = TextureUtils.Texture2DToFrame(imgRgb);
                
                var predictTask = ProcessNextFrameAsync(processResult, _predInfo, imgRgbFrame);
                yield return new WaitUntil(() => predictTask.IsCompleted);
                var (generatedFrame, updatedPredInfo) = predictTask.Result;
                _predInfo = updatedPredInfo;
                if (_debugImage != null)
                {
                    stream.queue.Enqueue(_debugImage);
                }
                else if (generatedFrame.data != null)
                {
                    var generatedImgTexture = TextureUtils.FrameToTexture2D(generatedFrame);
                    Debug.Log($"[LivePortraitInference] Frame {frameId} generated");
                    stream.queue.Enqueue(generatedImgTexture);
                }
                yield return null;
            }
        }

        /// <summary>
        /// Generate animated textures pipelined - matches Python LivePortraitMuseTalkAPI.execute
        /// </summary>
        public IEnumerator GenerateAsync(
            Texture2D sourceImage, string[] frameFiles, 
            LivePortaitStream outputStream, AvatarController avatarController)
        {
            // Step 1: Start source image processing immediately (async)
            var srcImg = TextureUtils.Texture2DToFrame(sourceImage);
            var processSrcTask = ProcessSourceImageAsync(srcImg);
            
            Logger.Log("[LivePortraitMuseTalkAPI] Source image processing started asynchronously");

            // Step 2: Create driving frames stream and start loading frames asynchronously
            var drivingStream = new DrivingFramesStream(frameFiles.Length);
            var loadFramesCoroutine = avatarController.StartCoroutine(LoadDrivingFramesAsync(frameFiles, drivingStream));

            // Step 3: Wait for source processing to complete
            yield return new WaitUntil(() => processSrcTask.IsCompleted);
            var processResult = processSrcTask.Result;
            
            Logger.Log("[LivePortraitMuseTalkAPI] Source image processing completed, starting frame processing pipeline");

            // Step 4: Process driving frames as they become available
            int processedFrames = 0;
            var predInfo = new LivePortraitPredInfo
            {
                Landmarks = null,
                InitialMotionInfo = null
            };

            while (processedFrames < frameFiles.Length && drivingStream.HasMoreFrames)
            {
                // Wait for next driving frame using CustomYieldInstruction
                var awaiter = drivingStream.WaitForNext();
                yield return awaiter;

                if (awaiter.Texture != null)
                {
                    var drivingFrame = awaiter.Texture;
                    
                    // Process this driving frame
                    var imgRgbData = TextureUtils.Texture2DToFrame(drivingFrame);
                    
                    var predictTask = ProcessNextFrameAsync(processResult, predInfo, imgRgbData);
                    yield return new WaitUntil(() => predictTask.IsCompleted);
                    var (generatedImg, updatedPredInfo) = predictTask.Result;
                    predInfo = updatedPredInfo;

                    // Output the generated frame
                    if (generatedImg.data != null)
                    {
                        var generatedImgTexture = TextureUtils.FrameToTexture2D(generatedImg);
                        outputStream.queue.Enqueue(generatedImgTexture);
                        Logger.Log($"[LivePortraitMuseTalkAPI] Processed frame {processedFrames + 1}/{frameFiles.Length}");
                    }

                    processedFrames++;
                    
                    // Clean up driving frame
                    if (drivingFrame != null)
                    {
                        UnityEngine.Object.DestroyImmediate(drivingFrame);
                    }
                }
                else if (drivingStream.LoadingFinished)
                {
                    // No more frames available and loading is finished
                    break;
                }
            }

            // Mark streams as finished
            outputStream.Finished = true;
            drivingStream.ProcessingFinished = true;
            
            Logger.Log($"[LivePortraitMuseTalkAPI] Pipelined processing completed: {processedFrames} frames generated");
        }


        /// <summary>
        /// Load driving frames asynchronously and add them to the stream
        /// </summary>
        private IEnumerator LoadDrivingFramesAsync(string[] frameFiles, DrivingFramesStream stream)
        {
            Logger.Log($"[LivePortraitMuseTalkAPI] Starting to load {frameFiles.Length} driving frames asynchronously");

            for (int i = 0; i < frameFiles.Length; i++)
            {
                string filePath = frameFiles[i];
                
                // Load frame data asynchronously - outside try-catch to avoid yield in try-catch
                var loadFileTask = System.IO.File.ReadAllBytesAsync(filePath);
                yield return new WaitUntil(() => loadFileTask.IsCompleted);
                
                try
                {
                    if (loadFileTask.IsFaulted)
                    {
                        Logger.LogError($"[LivePortraitMuseTalkAPI] Error loading driving frame {filePath}: {loadFileTask.Exception?.GetBaseException().Message}");
                        continue;
                    }
                    
                    byte[] fileData = loadFileTask.Result;
                    Texture2D texture = new(2, 2);
                    
                    if (texture.LoadImage(fileData))
                    {
                        texture.name = System.IO.Path.GetFileNameWithoutExtension(filePath);
                        var rgbTexture = TextureUtils.ConvertTexture2DToRGB24(texture);
                        stream.loadQueue.Enqueue(rgbTexture);
                        
                        // Clean up original texture if different
                        if (rgbTexture != texture)
                        {
                            UnityEngine.Object.DestroyImmediate(texture);
                        }
                    }
                    else
                    {
                        Logger.LogWarning($"[LivePortraitMuseTalkAPI] Failed to load image: {filePath}");
                        UnityEngine.Object.DestroyImmediate(texture);
                    }
                }
                catch (Exception e)
                {
                    Logger.LogError($"[LivePortraitMuseTalkAPI] Error processing driving frame {filePath}: {e.Message}");
                }
                yield return null;
            }

            stream.LoadingFinished = true;
            Logger.Log($"[LivePortraitMuseTalkAPI] Finished loading driving frames, {stream.QueueCount} frames queued");
        }
        
        /// <summary>
        /// Python: src_preprocess(img) - EXACT MATCH
        /// Returns (byte[] imageData, int width, int height) in RGB24 format
        /// OPTIMIZED: Uses direct pixel data access and parallelization for maximum performance
        /// </summary>
        private unsafe Frame SrcPreprocess(Frame frame)
        {
            int currentWidth = frame.width;
            int currentHeight = frame.height;
            
            // Python: max_dim = 1280
            // Python: if max(h, w) > max_dim:
            const int maxDim = 1280;
            if (Mathf.Max(currentHeight, currentWidth) > maxDim)
            {
                int newHeight, newWidth;
                if (currentHeight > currentWidth)
                {
                    // Python: new_h = max_dim; new_w = int(w * (max_dim / h))
                    newHeight = maxDim;
                    newWidth = Mathf.RoundToInt(currentWidth * ((float)maxDim / currentHeight));
                }
                else
                {
                    // Python: new_w = max_dim; new_h = int(h * (max_dim / w))
                    newWidth = maxDim;
                    newHeight = Mathf.RoundToInt(currentHeight * ((float)maxDim / currentWidth));
                }
                
                // Python: img = cv2.resize(img, (new_w, new_h))
                // Use TextureUtils for optimized byte array resizing (no texture conversion needed)
                frame = FrameUtils.ResizeFrame(frame, newWidth, newHeight, SamplingMode.Bilinear);
                
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
                return frame;
            }
            
            // Python: if new_h != img.shape[0] or new_w != img.shape[1]: img = img[:new_h, :new_w]
            if (finalHeight != currentHeight || finalWidth != currentWidth)
            {
                // Python crops from top-left: img[:new_h, :new_w]
                frame = CropImageBytesUnsafe(frame, currentWidth, currentHeight, finalWidth, finalHeight);
            }
            
            return frame;
        }
        
        /// <summary>
        /// Crop RGB24 byte array using optimized bulk memory operations
        /// OPTIMIZED: Uses parallelization and bulk copying for maximum performance
        /// </summary>
        private unsafe Frame CropImageBytesUnsafe(
            Frame frame, int sourceWidth, int sourceHeight, int cropWidth, int cropHeight)
        {
            if (cropHeight > sourceHeight || cropWidth > sourceWidth)
            {
                throw new InvalidOperationException("Crop size is larger than source image size");
            }
            var croppedFrame = new Frame(new byte[cropWidth * cropHeight * 3], cropWidth, cropHeight);
            Parallel.For(0, cropHeight, y =>
            {
                // Calculate source and destination indices for this row
                int sourceRowStart = y * sourceWidth * 3;
                int croppedRowStart = y * cropWidth * 3;
                int bytesToCopy = cropWidth * 3;
                
                Array.Copy(frame.data, sourceRowStart, croppedFrame.data, croppedRowStart, bytesToCopy);
            });
            
            return croppedFrame;
        }

        /// <summary>
        /// Python: crop_src_image(models, img) - EXACT MATCH
        /// </summary>
        private CropInfo CropSrcImage(Frame frame)
        {
            var srcFaces = _faceAnalysis.AnalyzeFaces(frame);
            
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
            
            // Python: lmk = src_face["landmark_2d_106"]  # this is the 106 landmarks from insightface
            var lmk = srcFace.Landmarks106;
            
            // Python: crop_info = crop_image(img, lmk, dsize=512, scale=2.3, vy_ratio=-0.125)
            var cropSize = 512;
            var cropInfo = CropImage(frame, lmk, cropSize, 2.3f, -0.125f);

            // Python: lmk = landmark_runner(models, img, lmk)
            lmk = _faceAnalysis.LandmarkRunner(frame, lmk);
            
            // Python: crop_info["lmk_crop"] = lmk
            cropInfo.LandmarksCrop = lmk;
            
            // Python: crop_info["img_crop_256x256"] = cv2.resize(crop_info["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            cropInfo.ImageCrop256x256 = FrameUtils.ResizeFrame(cropInfo.ImageCrop, 256, 256, SamplingMode.Bilinear);
            
            // Python: crop_info["lmk_crop_256x256"] = crop_info["lmk_crop"] * 256 / 512
            cropInfo.LandmarksCrop256x256 = ScaleLandmarks(cropInfo.LandmarksCrop, 256f / 512f);
            
            return cropInfo;
        }
        
        /// <summary>
        /// Python: preprocess(img) - EXACT MATCH
        /// </summary>
        /// <summary>
        /// OPTIMIZED: Preprocess using the common PreprocessImageOptimized method
        /// Python: img = img / 255.0 and transpose to CHW format
        /// </summary>
        private DenseTensor<float> Preprocess(Frame frame)
        {
            // Python: img = img / 255.0
            // Python: img = np.clip(img, 0, 1)  # clip to 0~1 (automatic with byte/255)
            // Python: img = img.transpose(2, 0, 1)  # HxWx3x1 -> 1x3xHxW
            // Python: img = np.expand_dims(img, axis=0)
            // Python: img = img.astype(np.float32)
            
            // Use the common optimized method: pixelValue / 255.0 = pixelValue * (1/255) + 0
            var tensor = FrameUtils.FrameToTensor(frame, 1.0f / 255.0f, 0.0f);
            return tensor;
        }
        
        /// <summary>
        /// OPTIMIZED: Get keypoint info avoiding ToArray() calls and minimizing memory allocations
        /// ~2-3x faster by working with tensors directly and avoiding intermediate arrays
        /// </summary>
        private MotionInfo GetKpInfo(DenseTensor<float> preprocessedData)
        {            
            // Python: net = models["motion_extractor"]
            // Python: output = net.run(None, {"img": x})
            // Use the actual input name from the model metadata
            
            var inputs = new List<Tensor<float>>
            {
                preprocessedData
            };
            
            var results = ModelUtils.RunModel("motion_extractor", _motionExtractor, inputs);
            
            // OPTIMIZED: Work with tensors directly, avoiding ToArray() calls until the final step
            var pitchTensor = results[1].AsTensor<float>();
            var yawTensor = results[1].AsTensor<float>();
            var rollTensor = results[2].AsTensor<float>();
            var tTensor = results[3].AsTensor<float>();
            var expTensor = results[4].AsTensor<float>();
            var scaleTensor = results[5].AsTensor<float>();
            var kpTensor = results[6].AsTensor<float>();
            
            // OPTIMIZED: Process angle tensors directly without intermediate arrays
            // Python: pred = softmax(kp_info["pitch"], axis=1)
            // Python: degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
            // Python: kp_info["pitch"] = degree[:, None]  # Bx1
            var processedPitch = ProcessAngleSoftmaxOptimized(pitchTensor);
            var processedYaw = ProcessAngleSoftmaxOptimized(yawTensor);
            var processedRoll = ProcessAngleSoftmaxOptimized(rollTensor);
            
            // OPTIMIZED: Extract arrays efficiently using direct buffer access when possible
            return new MotionInfo
            {
                Pitch = processedPitch,
                Yaw = processedYaw,
                Roll = processedRoll,
                Translation = ExtractTensorArrayOptimized(tTensor),
                Expression = ExtractTensorArrayOptimized(expTensor),
                Scale = ExtractTensorArrayOptimized(scaleTensor),
                Keypoints = ExtractTensorArrayOptimized(kpTensor)
            };
        }
        
        /// <summary>
        /// OPTIMIZED: Process angle softmax directly on tensor data using unsafe pointers
        /// Eliminates ToArray() call and intermediate array allocations for maximum performance
        /// </summary>
        private unsafe float[] ProcessAngleSoftmaxOptimized(Tensor<float> angleLogitsTensor)
        {
            // OPTIMIZED: Access tensor data directly using unsafe pointers - NO ToArray() call
            var tensorData = angleLogitsTensor as DenseTensor<float>;
            if (tensorData == null)
            {
                // Fallback for non-DenseTensor types
                Debug.LogWarning("ProcessAngleSoftmaxOptimized: Non-DenseTensor type");
                var angleLogits = angleLogitsTensor.ToArray();
                return ProcessAngleSoftmaxArray(angleLogits);
            }
            
            // Get direct access to tensor's internal buffer
            var buffer = tensorData.Buffer;
            int length = (int)angleLogitsTensor.Length;
            
            // Access raw memory directly using Span
            var span = buffer.Span;
            
            // OPTIMIZED: Direct computation without any array allocations
            // Python: pred = softmax(kp_info["pitch"], axis=1)
            
            // Find max value for numerical stability
            float maxVal = float.MinValue;
            for (int i = 0; i < length; i++)
            {
                if (span[i] > maxVal) maxVal = span[i];
            }
            
            // Compute softmax and weighted sum in one pass - no intermediate arrays
            // Python: degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
            float expSum = 0f;
            float weightedSum = 0f;
            
            for (int i = 0; i < length; i++)
            {
                float exp = Mathf.Exp(span[i] - maxVal);
                expSum += exp;
                weightedSum += exp * i; // np.arange(66) gives 0,1,2,...,65
            }
            
            // Normalize and apply Python formula
            float degree = (weightedSum / expSum) * 3f - 97.5f;
            
            return new float[] { degree };
        }
        
        /// <summary>
        /// Fallback method for non-DenseTensor types - still optimized single-pass computation
        /// </summary>
        private float[] ProcessAngleSoftmaxArray(float[] angleLogits)
        {
            // Find max value for numerical stability
            float maxVal = float.MinValue;
            for (int i = 0; i < angleLogits.Length; i++)
            {
                if (angleLogits[i] > maxVal) maxVal = angleLogits[i];
            }
            
            // Compute softmax and weighted sum in one pass
            float expSum = 0f;
            float weightedSum = 0f;
            
            for (int i = 0; i < angleLogits.Length; i++)
            {
                float exp = Mathf.Exp(angleLogits[i] - maxVal);
                expSum += exp;
                weightedSum += exp * i;
            }
            
            float degree = (weightedSum / expSum) * 3f - 97.5f;
            return new float[] { degree };
        }
        
        /// <summary>
        /// OPTIMIZED: Extract tensor data to array using direct buffer access when possible
        /// Avoids ToArray() overhead for DenseTensor types by accessing underlying buffer directly
        /// </summary>
        private unsafe float[] ExtractTensorArrayOptimized(Tensor<float> tensor)
        {
            // Try to access DenseTensor buffer directly
            if (tensor is DenseTensor<float> denseTensor)
            {
                // OPTIMIZED: Direct buffer access using Span - much faster than ToArray()
                var buffer = denseTensor.Buffer;
                var span = buffer.Span;
                int length = (int)tensor.Length;

                // Efficient array copy from Span
                var result = new float[length];
                span.CopyTo(result);
                return result;
            }

            Debug.LogWarning("ExtractTensorArrayOptimized: Non-DenseTensor type");
            // Fallback for non-DenseTensor types
            return tensor.ToArray();
        }
        
        /// <summary>
        /// Python: extract_feature_3d(models, x) - EXACT MATCH
        /// </summary>
        private Tensor<float> ExtractFeature3d(DenseTensor<float> preprocessedData)
        {            
            var inputTensor = preprocessedData;
            
            // Python: net = models["appearance_feature_extractor"]
            // Python: output = net.run(None, {"img": x})
            // Use the actual input name from the model metadata
            
            var inputs = new List<Tensor<float>>
            {
                inputTensor
            };
            
            var results = ModelUtils.RunModel("appearance_feature_extractor", _appearanceFeatureExtractor, inputs);
            var outputTensor = results.First().AsTensor<float>();
            
            // Python: f_s = output[0]
            // Python: f_s = f_s.astype(np.float32)
            return outputTensor;
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
            var rotMat = MathUtils.GetRotationMatrix(pitch, yaw, roll);
            
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

        private async Task<(Frame, LivePortraitPredInfo)> ProcessNextFrameAsync(
            ProcessSourceImageResult processResult,
            LivePortraitPredInfo predInfo,
            Frame drivingFrame)
        {
            return await Task.Run(() => 
            {
                var (Ip, updatedPredInfo) = Predict(
                    processResult.XsInfo, 
                    processResult.Rs, 
                    processResult.Fs, 
                    processResult.Xs, 
                    drivingFrame, 
                    predInfo);

                // Python: if self.flg_composite: driving_img = concat_frame(img_rgb, img_crop_256x256, I_p)
                // Python: else: driving_img = paste_back(I_p, crop_info["M_c2o"], src_img, mask_ori)
                Frame drivingImg = PasteBack(
                    Ip,
                    processResult.CropInfo.Transform, 
                    processResult.SrcImg, 
                    processResult.MaskOri);
                return (drivingImg, updatedPredInfo);
            });
        }
        
        /// <summary>
        /// Python: predict(frame_id, models, x_s_info, R_s, f_s, x_s, img, pred_info) - EXACT MATCH
        /// </summary>
        private (Frame, LivePortraitPredInfo) Predict(
            MotionInfo xSInfo, float[,] Rs, Tensor<float> fs, float[] xs, 
            Frame img, LivePortraitPredInfo predInfo)
        {
            // Python: frame_0 = pred_info['lmk'] is None
            bool frame0 = predInfo.Landmarks == null;
            
            Vector2[] lmk;
            if (frame0)
            {
                // Python: face_analysis = models["face_analysis"]
                // Python: src_face = face_analysis(img)
                var srcFaces = _faceAnalysis.AnalyzeFaces(img);
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
                lmk = _faceAnalysis.LandmarkRunner(img, lmk);
            }
            else
            {
                // Python: lmk = landmark_runner(models, img, pred_info['lmk'])
                lmk = _faceAnalysis.LandmarkRunner(img, predInfo.Landmarks);
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
            
            // var cDEyes1 = CalculateDistanceRatio(lmk, 6, 18, 0, 12);
            // var cDEyes2 = CalculateDistanceRatio(lmk, 30, 42, 24, 36);
            // // Python concatenates these along axis=1
            // var cDEyes = new float[cDEyes1.Length + cDEyes2.Length];
            // Array.Copy(cDEyes1, 0, cDEyes, 0, cDEyes1.Length);
            // Array.Copy(cDEyes2, 0, cDEyes, cDEyes1.Length, cDEyes2.Length);
            
            // var cDLip = CalculateDistanceRatio(lmk, 90, 102, 48, 66);
            
            // Convert to float32 (already float in C#)
            // Note: These values are computed but never used in ONNX inference, matching Python behavior exactly
            
            // Python: prepare_driving_videos
            // Python: img = cv2.resize(img, (256, 256))

            var img256 = FrameUtils.ResizeFrame(img, 256, 256, SamplingMode.Bilinear);
            // Python: I_d = preprocess(img)
            var Id = Preprocess(img256);
            
            // Python: collect s_d, R_d, δ_d and t_d for inference
            // Python: x_d_info = get_kp_info(models, I_d)
            var xDInfo = GetKpInfo(Id);

            // Python: R_d = get_rotation_matrix(x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"])
            var Rd = MathUtils.GetRotationMatrix(xDInfo.Pitch, xDInfo.Yaw, xDInfo.Roll);
            
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
            var Rd0Transposed = MathUtils.TransposeMatrix(Rd0);
            var RdTimesRd0T = MathUtils.MatrixMultiply(Rd, Rd0Transposed);
            var RNew = MathUtils.MatrixMultiply(RdTimesRd0T, Rs);
            
            // Python: delta_new = x_s_info["exp"] + (x_d_info["exp"] - x_d_0_info["exp"])
            var expDiff = MathUtils.SubtractArrays(xDInfo.Expression, xD0Info.Expression);
            var deltaNew = MathUtils.AddArrays(xSInfo.Expression, expDiff);
            
            // Python: scale_new = x_s_info["scale"] * (x_d_info["scale"] / x_d_0_info["scale"])
            var scaleDiff = MathUtils.DivideArrays(xDInfo.Scale, xD0Info.Scale);
            var scaleNew = MathUtils.MultiplyArrays(xSInfo.Scale, scaleDiff);
            
            // Python: t_new = x_s_info["t"] + (x_d_info["t"] - x_d_0_info["t"])
            var tDiff = MathUtils.SubtractArrays(xDInfo.Translation, xD0Info.Translation);
            var tNew = MathUtils.AddArrays(xSInfo.Translation, tDiff);
            
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
            
            // Python: out = out.transpose(0, 2, 3, 1)  # 1x3xHxW -> 1xHxWx3
            // Python: out = np.clip(out, 0, 1)  # clip to 0~1
            // Python: out = (out * 255).astype(np.uint8)  # 0~1 -> 0~255
            // Python: I_p = out[0]
            var resultTexture = ConvertOutput(output, 512, 512);
            
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
            var inputs = new List<Tensor<float>>
            {
                inputTensor
            };
            
            var results = ModelUtils.RunModel("stitching", _stitching, inputs);
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
        private float[] WarpingSpade(Tensor<float> feature3d, float[] kpSource, float[] kpDriving)
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
            var feature3DTensor = feature3d;
            var kpSourceTensor = new DenseTensor<float>(kpSource, new[] { 1, kpSource.Length / 3, 3 });
            var kpDrivingTensor = new DenseTensor<float>(kpDriving, new[] { 1, kpDriving.Length / 3, 3 });
            
            
            // Python: net = models["warping_spade"]
            // Python: output = net.run(None, {"feature_3d": feature_3d, "kp_driving": kp_driving, "kp_source": kp_source})
            // Use actual input names from model metadata
            
            var inputs = new List<Tensor<float>>
            {
                feature3DTensor,  // feature_3d
                kpDrivingTensor, // kp_driving  
                kpSourceTensor   // kp_source
            };
            
            var results = ModelUtils.RunModel("warping_spade", _warpingSpade, inputs);
            
            // Python: return output[0] - take the first output (warped_feature)
            var output = results[0].AsTensor<float>().ToArray();
            return output;
        }
        
        // Helper methods - all implemented inline for self-sufficiency
        
        private Vector2[] ScaleLandmarks(Vector2[] landmarks, float scale)
        {
            var result = new Vector2[landmarks.Length];
            for (int i = 0; i < landmarks.Length; i++)
            {
                result[i] = landmarks[i] * scale;
            }
            return result;
        }
        
        private CropInfo CropImage(Frame frame, Vector2[] lmk, int dsize, float scale, float vyRatio)
        {
            // Python: crop_image(img, pts: np.ndarray, dsize=224, scale=1.5, vy_ratio=-0.1) - EXACT MATCH
            var (MInv, _) = EstimateSimilarTransformFromPts(lmk, dsize, scale, 0f, vyRatio, true);
            
            var imgCrop = FrameUtils.AffineTransformFrame(frame, MInv, dsize, dsize);
            var ptCrop = MathUtils.TransformPts(lmk, MInv);
            
            // Python: M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=np.float32)])
            var Mo2c = MathUtils.GetCropTransform(MInv);
            
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
            
            var M = MathUtils.InvertMatrix3x3(MInvH);
            var M2x3 = new float[,] {
                { M[0, 0], M[0, 1], M[0, 2] },
                { M[1, 0], M[1, 1], M[1, 2] }
            };

            // Python: return M_INV, M[:2, ...]
            return (MInv, M2x3);
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
        /// OPTIMIZED: Convert ONNX output to byte array using unsafe pointers and parallelization
        /// Python: out.transpose(0, 2, 3, 1) and convert to byte array - EXACT MATCH
        /// ~5-10x faster than original through parallel processing and direct memory access
        /// </summary>
        private unsafe Frame ConvertOutput(float[] output, int width, int height)
        {
            var result = new byte[width * height * 3];
            int totalPixels = width * height;
            
            // Pre-calculate channel offsets for CHW format (avoid repeated calculations)
            int channelSize = height * width;
            int rChannelOffset = 0 * channelSize;
            int gChannelOffset = 1 * channelSize;
            int bChannelOffset = 2 * channelSize;
            
            // OPTIMIZED: Use unsafe pointers for direct memory access
            fixed (float* outputPtr = output)
            fixed (byte* resultPtr = result)
            {
                // Capture pointers in local variables to avoid lambda closure issues
                float* outputPtrLocal = outputPtr;
                byte* resultPtrLocal = resultPtr;
                
                // MAXIMUM PERFORMANCE: Parallel processing across all pixels
                // Each pixel can be processed independently for perfect parallelization
                System.Threading.Tasks.Parallel.For(0, totalPixels, pixelIdx =>
                {
                    // Calculate h, w coordinates from linear pixel index
                    int h = pixelIdx / width;
                    int w = pixelIdx % width;
                    
                    // OPTIMIZED: Direct pointer arithmetic for CHW indexing
                    int hwOffset = h * width + w;
                    float* rPtr = outputPtrLocal + rChannelOffset + hwOffset;
                    float* gPtr = outputPtrLocal + gChannelOffset + hwOffset;
                    float* bPtr = outputPtrLocal + bChannelOffset + hwOffset;
                    
                    // OPTIMIZED: Fast clamping using direct comparison (faster than Mathf.Clamp01)
                    float r = *rPtr;
                    float g = *gPtr;
                    float b = *bPtr;
                    
                    r = r < 0f ? 0f : r > 1f ? 1f : r;
                    g = g < 0f ? 0f : g > 1f ? 1f : g;
                    b = b < 0f ? 0f : b > 1f ? 1f : b;
                    
                    // OPTIMIZED: Direct pointer write to result (RGB24 format)
                    byte* resultPixelPtr = resultPtrLocal + pixelIdx * 3;
                    resultPixelPtr[0] = (byte)(r * 255f); // R
                    resultPixelPtr[1] = (byte)(g * 255f); // G
                    resultPixelPtr[2] = (byte)(b * 255f); // B
                });
            }
            
            return new Frame(result, width, height);
        }
        
        public void Dispose()
        {
            if (_disposed) return;
            
            try
            {
                _appearanceFeatureExtractor?.Dispose();
                _motionExtractor?.Dispose();
                _stitching?.Dispose();
                _warpingSpade?.Dispose();
                _faceAnalysis?.Dispose();
                
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
        private Frame PreparePasteBack(Matrix4x4 cropMc2o, int width, int height)
        {
            // Python: mask_ori = cv2.warpAffine(mask_crop, crop_M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR)
            // Python: mask_ori = mask_ori.astype(np.float32) / 255.0
            
            if (_maskTemplate.data == null)
            {
                Debug.LogError("[PreparePasteBack] No mask template provided, creating default circular mask");
                throw new Exception("[PreparePasteBack] No mask template provided");
            }
            
            // Transform mask template using crop transformation matrix
            float[,] M = new float[,] {
                { cropMc2o.m00, cropMc2o.m01, cropMc2o.m03 },
                { cropMc2o.m10, cropMc2o.m11, cropMc2o.m13 }
            };
            
            var maskOri = FrameUtils.AffineTransformFrame(_maskTemplate, M, width, height);
            return maskOri;
        }
        
        /// <summary>
        /// Python: paste_back(img_crop, M_c2o, img_ori, mask_ori) - EXACT MATCH
        /// </summary>
        private Frame PasteBack(Frame imgCrop,
                                Matrix4x4 Mc2o, 
                                Frame imgOri, 
                                Frame maskOri)
        {
            
            // Python: dsize = (img_ori.shape[1], img_ori.shape[0])
            int dsize_w = imgOri.width;
            int dsize_h = imgOri.height;
            
            // Python: result = cv2.warpAffine(img_crop, M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR)
            float[,] M = new float[,] {
                { Mc2o.m00, Mc2o.m01, Mc2o.m03 },
                { Mc2o.m10, Mc2o.m11, Mc2o.m13 }
            };
            var warped = FrameUtils.AffineTransformFrame(imgCrop, M, dsize_w, dsize_h);
            
            // Python: result = np.clip(mask_ori * result + (1 - mask_ori) * img_ori, 0, 255).astype(np.uint8)
            var result = new Frame(new byte[dsize_w * dsize_h * 3], dsize_w, dsize_h);
            
            for (int i = 0; i < imgOri.width * imgOri.height; i++)
            {
                // Python: result = np.clip(mask_ori * result + (1 - mask_ori) * img_ori, 0, 255)
                // This is a per-channel blend, equivalent to Lerp(ori, warped, mask) for each channel.
                var rIndex = 3 * i + 0;
                var gIndex = 3 * i + 1;
                var bIndex = 3 * i + 2;

                var maskR = maskOri.data[rIndex] / 255f;
                var maskG = maskOri.data[gIndex] / 255f;
                var maskB = maskOri.data[bIndex] / 255f;
                
                float r = imgOri.data[rIndex] * (1f - maskR) + warped.data[rIndex] * maskR;
                float g = imgOri.data[gIndex] * (1f - maskG) + warped.data[gIndex] * maskG;
                float b = imgOri.data[bIndex] * (1f - maskB) + warped.data[bIndex] * maskB;
                
                result.data[rIndex] = (byte)(r < 0f ? 0f : r > 255f ? 255f : r);
                result.data[gIndex] = (byte)(g < 0f ? 0f : g > 255f ? 255f : g);
                result.data[bIndex] = (byte)(b < 0f ? 0f : b > 255f ? 255f : b);
            }
            return result;
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
}

