using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LiveTalk.Core
{
    using API;
    using Utils;


    /// <summary>
    /// Input for MuseTalk inference - simplified for streaming
    /// </summary>
    internal class MuseTalkInput
    {
        /// <summary>
        /// Avatar images for talking head generation
        /// </summary>
        public Texture2D[] AvatarTextures { get; set; }
        
        /// <summary>
        /// Audio clip for lip sync
        /// </summary>
        public AudioClip AudioClip { get; set; }
        
        /// <summary>
        /// Batch size for processing
        /// </summary>
        public int BatchSize { get; set; } = 4;
        
        public MuseTalkInput(Texture2D avatarTexture, AudioClip audioClip)
        {
            AvatarTextures = new[] { avatarTexture ?? throw new ArgumentNullException(nameof(avatarTexture)) };
            AudioClip = audioClip ?? throw new ArgumentNullException(nameof(audioClip));
        }
        
        public MuseTalkInput(Texture2D[] avatarTextures, AudioClip audioClip)
        {
            AvatarTextures = avatarTextures ?? throw new ArgumentNullException(nameof(avatarTextures));
            AudioClip = audioClip ?? throw new ArgumentNullException(nameof(audioClip));
        }
    }

    /// <summary>
    /// Avatar animation cache key based on texture content hashes
    /// </summary>
    internal class AvatarAnimationKey
    {
        public string[] TextureHashes { get; set; }
        public string Version { get; set; }
        
        public override bool Equals(object obj)
        {
            if (obj is AvatarAnimationKey other)
            {
                return Version == other.Version && 
                       TextureHashes != null && other.TextureHashes != null &&
                       TextureHashes.Length == other.TextureHashes.Length &&
                       TextureHashes.SequenceEqual(other.TextureHashes);
            }
            return false;
        }
        
        public override int GetHashCode()
        {
            unchecked
            {
                int hash = Version?.GetHashCode() ?? 0;
                if (TextureHashes != null)
                {
                    foreach (var textureHash in TextureHashes)
                    {
                        hash = hash * 31 + (textureHash?.GetHashCode() ?? 0);
                    }
                }
                return hash;
            }
        }
    }

    /// <summary>
    /// Core MuseTalk inference engine that manages ONNX models for real-time talking head generation
    /// 
    /// MEMORY OPTIMIZATIONS:
    /// - Zero-copy tensor operations in UNet and VAE decoder pipeline
    /// - Reusable batch arrays to eliminate repeated allocations  
    /// - Direct buffer sharing between ONNX tensors without intermediate copying
    /// - Persistent ONNX result storage to prevent premature garbage collection
    /// - Avatar animation caching for repeated sequences
    /// </summary>
    internal class MuseTalkInference : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        // ONNX Runtime sessions
        private InferenceSession _unetSession;
        private InferenceSession _vaeEncoderSession;
        private InferenceSession _vaeDecoderSession;
        private InferenceSession _positionalEncodingSession;
        
        // Whisper model for audio feature extraction
        private readonly WhisperModel _whisperModel;
        
        // Configuration
        private LiveTalkConfig _config;
        private readonly bool _initialized = false;
        private bool _disposed = false;
        
        // QUANTIZATION SUPPORT: INT8 configuration for CPU optimization
        private readonly bool _useINT8 = false;
        
        // Face analysis utility for SCRFD+1k3d68 face processing
        private readonly FaceAnalysis _faceAnalysis;
        
        // Avatar data for blending
        private AvatarData _avatarData;
        
        // Avatar animation segmentation cache
        private static readonly Dictionary<AvatarAnimationKey, AvatarData> _avatarAnimationCache = new();
        private const int MAX_CACHE_SIZE = 1000; // Limit cache size to prevent memory bloat
        
        // Disk cache for persistent avatar processing results
        private readonly AvatarDiskCache _diskCache;
        
        // Reusable batch processing array (class-level for memory reuse)
        private float[] _reusableBatchArray = new float[0];
        
        // Reusable UNet result to keep tensor memory alive across pipeline
        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _reusableUNetResult;
        
        public bool IsInitialized => _initialized;
        public static bool LogTiming { get; set; } = false;
        
        /// <summary>
        /// Enable detailed performance monitoring and logging
        /// </summary>
        public static bool EnablePerformanceMonitoring { get; set; } = true;
        
        /// <summary>
        /// Returns true if INT8 quantization is currently active (excluding VAE models for quality)
        /// </summary>
        public bool IsUsingINT8 => _useINT8;
        
        /// <summary>
        /// Get the current quantization mode being used
        /// </summary>
        public string QuantizationMode
        {
            get
            {
                if (_useINT8) return "Optimal (INT8 for performance, FP32 for VAE quality)";
                return "FP32 only";
            }
        }
        
        /// <summary>
        /// Initialize MuseTalk inference with specified configuration
        /// </summary>
        public MuseTalkInference(LiveTalkConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            
            // QUANTIZATION: Initialize INT8 support based on config
            _useINT8 = _config.UseINT8;
            
            try
            {
                InitializeModels();
                _faceAnalysis = new FaceAnalysis(_config);
                
                // Initialize Whisper model for audio processing
                _whisperModel = new WhisperModel(_config);
                
                // Initialize disk cache for persistent avatar processing results
                try
                {
                    _diskCache = new AvatarDiskCache(_config);
                    Logger.Log("[MuseTalkInference] Disk cache initialized successfully");
                }
                catch (Exception diskCacheEx)
                {
                    Logger.LogWarning($"[MuseTalkInference] Failed to initialize disk cache: {diskCacheEx.Message}. Proceeding without disk caching.");
                    // _diskCache remains null, will be handled gracefully in code
                }
                
                _initialized = true;
                Logger.Log("[MuseTalkInference] Successfully initialized");
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkInference] Failed to initialize: {e.Message}");
                _initialized = false;
            }
        }
        
        /// <summary>
        /// Initialize all ONNX models
        /// </summary>
        private void InitializeModels()
        {
            _unetSession = ModelUtils.LoadModel(_config, "unet");
            _vaeEncoderSession = ModelUtils.LoadModel(_config, "vae_encoder");
            _vaeDecoderSession = ModelUtils.LoadModel(_config, "vae_decoder");
            _positionalEncodingSession = ModelUtils.LoadModel(_config, "positional_encoding");
        }

        /// <summary>
        /// Generate talking head video frames from avatar images and audio (STREAMING)
        /// Matches LivePortrait's streaming approach - yields frames as they're generated
        /// MAIN THREAD ONLY for correctness
        /// </summary>
        public IEnumerator GenerateAsync(MuseTalkInput input, OutputStream stream)
        {
            if (!_initialized)
                throw new InvalidOperationException("MuseTalk inference not initialized");
                
            if (input == null)
                throw new ArgumentNullException(nameof(input));
                
            Logger.Log($"[MuseTalkInference] === STARTING MUSETALK STREAMING GENERATION ===");
            Logger.Log($"[MuseTalkInference] Version: {_config.Version}, Batch Size: {input.BatchSize}");
            Logger.Log($"[MuseTalkInference] Avatar Images: {input.AvatarTextures.Length}, Audio: {input.AudioClip.name} ({input.AudioClip.length:F2}s)");
            
            // Step 1: Process avatar images and extract face regions
            Logger.Log("[MuseTalkInference] STAGE 1: Processing avatar images...");
            var avatarTask = ProcessAvatarImages(input.AvatarTextures);
            yield return new WaitUntil(() => avatarTask.IsCompleted);
            var avatarData = avatarTask.Result;
            Logger.Log($"[MuseTalkInference] Stage 1 completed - Processed {avatarData.FaceRegions.Count} faces");
            
            // Step 2: Process audio and extract features
            Logger.Log("[MuseTalkInference] STAGE 2: Processing audio...");
            var audioTask = ProcessAudio(input.AudioClip);
            yield return new WaitUntil(() => audioTask.IsCompleted);
            var audioFeatures = audioTask.Result;
            stream.TotalExpectedFrames = audioFeatures.FeatureChunks.Count;
            Logger.Log($"[MuseTalkInference] Stage 2 completed - Generated {audioFeatures.FeatureChunks.Count} audio chunks");
            
            // Step 3: Generate video frames in streaming mode
            Logger.Log("[MuseTalkInference] STAGE 3: Generating video frames (streaming)...");
            yield return GenerateFramesStreaming(avatarData, audioFeatures, input.BatchSize, stream);
            
            stream.Finished = true;
            Logger.Log($"[MuseTalkInference] === STREAMING GENERATION COMPLETED ===");
        }
        
        /// <summary>
        /// Pre-compute segmentation data that can be cached and reused for all frames
        /// This includes face_large crop, BiSeNet segmentation mask, and all blending masks
        /// REFACTORED: Uses byte arrays internally for better memory efficiency
        /// </summary>
        private SegmentationData PrecomputeSegmentationData(Frame originalImage, Vector4 faceBbox, string version)
        {
            // Apply version-specific adjustments to face bbox (matching BlendFaceWithOriginal logic)
            Vector4 adjustedFaceBbox = faceBbox;
            if (version == "v15") // v15 mode
            {
                // Apply v15 extra margin to y2 (bottom of face bbox)
                adjustedFaceBbox.w = Mathf.Min(adjustedFaceBbox.w + 10f, originalImage.height);
            }
            
            // Calculate expanded crop box for face_large (matching Python's crop_box calculation)
            var cropRect = GetCropBox(adjustedFaceBbox, 1.5f); // expandFactor = 1.5f
            var cropBox = new Vector4(cropRect.x, cropRect.y, cropRect.x + cropRect.width, cropRect.y + cropRect.height);
            
            // Create face_large crop (matching Python's face_large = body.crop(crop_box))
            var faceLarge = CropImage(originalImage, cropRect);
            
            // Generate face segmentation mask using BiSeNet on face_large
            var segmentationMask = GenerateFaceSegmentationMaskCached(faceLarge);
            
            if (segmentationMask.data == null)
                throw new InvalidOperationException("Failed to generate segmentation mask");
            
            // OPTIMIZATION: Precompute all blending masks that are independent of faceTexture
            // These masks only depend on segmentation, face bbox, and crop box - all available now
            
            // Step 1: Create mask_small by cropping BiSeNet mask to face bbox (matching Python)
            var maskSmall = ImageBlendingHelper.CreateSmallMask(segmentationMask, adjustedFaceBbox, cropRect);
            
            // Step 2: Create full mask by pasting mask_small back into face_large dimensions (matching Python)
            var fullMask = ImageBlendingHelper.CreateFullMask(segmentationMask, maskSmall, adjustedFaceBbox, cropRect);
            
            // Step 3: Apply upper boundary ratio to preserve upper face (matching Python)
            const float upperBoundaryRatio = 0.5f; // Standard value used in ApplySegmentationMask
            var boundaryMask = ImageBlendingHelper.ApplyUpperBoundaryRatio(fullMask, upperBoundaryRatio);
            
            // Step 4: Apply Gaussian blur for smooth blending (matching Python)
            var blurredMask = ImageBlendingHelper.ApplyGaussianBlurToMask(boundaryMask);
            
            return new SegmentationData
            {
                FaceLarge = faceLarge,
                SegmentationMask = segmentationMask,
                AdjustedFaceBbox = adjustedFaceBbox,
                CropBox = cropBox,
                MaskSmall = maskSmall,
                FullMask = fullMask,                
                BoundaryMask = boundaryMask,
                BlurredMask = blurredMask
            };
        }
        
        /// <summary>
        /// Generate segmentation mask for caching (extracted from ImageBlendingHelper logic)
        /// FIXED: Runs on main thread to avoid Unity texture operation violations
        /// </summary>
        private Frame GenerateFaceSegmentationMaskCached(Frame faceLarge)
        {
            try
            {
                // Run BiSeNet directly on the face_large crop using byte array optimized method
                var mask = _faceAnalysis.CreateFaceMaskWithMorphology(faceLarge, "jaw");
                
                if (mask.data != null)
                {
                    // Resize to target dimensions if needed
                    if (mask.width != faceLarge.width || mask.height != faceLarge.height)
                    {
                        var resizedMask = FrameUtils.ResizeFrame(mask, faceLarge.width, faceLarge.height, SamplingMode.Bilinear);
                        return resizedMask;
                    }
                    return mask;
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkInference] ONNX face parsing failed: {e.Message}");
            }
            
            throw new InvalidOperationException("Face segmentation failed and no fallback is available");
        }
        
        /// <summary>
        /// Calculate crop box with expansion factor (matching Python get_crop_box)
        /// </summary>
        private Rect GetCropBox(Vector4 faceBbox, float expandFactor)
        {
            // Python: x, y, x1, y1 = box
            float x = faceBbox.x;
            float y = faceBbox.y;
            float x1 = faceBbox.z;
            float y1 = faceBbox.w;
            
            // Python: x_c, y_c = (x+x1)//2, (y+y1)//2 (integer division!)
            int xCenter = (int)((x + x1) / 2);
            int yCenter = (int)((y + y1) / 2);
            
            // Python: w, h = x1-x, y1-y
            float width = x1 - x;
            float height = y1 - y;
            
            // Python: s = int(max(w, h)//2*expand) (integer conversion!)
            int s = (int)(Mathf.Max(width, height) / 2 * expandFactor);
            
            // Python: crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
            return new Rect(xCenter - s, yCenter - s, 2 * s, 2 * s);
        }
        
        /// <summary>
        /// Crop image to specified rectangle
        /// </summary>
        private Frame CropImage(Frame source, Rect cropRect)
        {
            // Ensure crop bounds are within image
            cropRect.x = Mathf.Max(0, cropRect.x);
            cropRect.y = Mathf.Max(0, cropRect.y);
            cropRect.width = Mathf.Min(cropRect.width, source.width - cropRect.x);
            cropRect.height = Mathf.Min(cropRect.height, source.height - cropRect.y);
            
            return FrameUtils.CropFrame(source, cropRect);
        }
        
        /// <summary>
        /// Create cache key for avatar animation sequence
        /// </summary>
        private AvatarAnimationKey CreateAvatarAnimationKey(Texture2D[] avatarTextures, string version)
        {
            var textureHashes = new string[avatarTextures.Length];
            for (int i = 0; i < avatarTextures.Length; i++)
            {
                textureHashes[i] = TextureUtils.GenerateTextureHash(avatarTextures[i]);
            }
            
            return new AvatarAnimationKey
            {
                TextureHashes = textureHashes,
                Version = version
            };
        }
        
        /// <summary>
        /// Manage cache size to prevent memory bloat
        /// </summary>
        private void ManageCacheSize()
        {
            if (_avatarAnimationCache.Count > MAX_CACHE_SIZE)
            {
                // Remove oldest entries (simple FIFO for now)
                var keysToRemove = _avatarAnimationCache.Keys.Take(_avatarAnimationCache.Count - MAX_CACHE_SIZE + 1).ToList();
                foreach (var key in keysToRemove)
                {
                    _avatarAnimationCache.Remove(key);
                    Logger.Log($"[MuseTalkInference] Removed cached avatar animation to manage memory");
                }
            }
        }

        /// <summary>
        /// Process avatar images and extract face regions with landmarks
        /// Matches Python's get_landmark_and_bbox functionality exactly
        /// OPTIMIZED: Uses avatar animation caching for repeated sequences
        /// ENHANCED: Added disk caching for persistent avatar processing results
        /// </summary>
        private async Task<AvatarData> ProcessAvatarImages(Texture2D[] avatarTextures)
        {
            // Generate cache keys for both in-memory and disk caching
            var memoryCacheKey = CreateAvatarAnimationKey(avatarTextures, _config.Version);
            var diskCacheKey = _diskCache?.GenerateAvatarCacheKey(avatarTextures, _config.Version);
            
            // Check in-memory cache first (fastest)
            if (_avatarAnimationCache.TryGetValue(memoryCacheKey, out var cachedAvatarData))
            {
                Logger.Log($"[MuseTalkInference] Using in-memory cached avatar animation data for {avatarTextures.Length} textures - {cachedAvatarData.FaceRegions.Count} faces, {cachedAvatarData.Latents.Count} latents");
                
                // Validate cached data integrity
                if (cachedAvatarData.Latents.Count == 0)
                {
                    Logger.LogWarning("[MuseTalkInference] Cached avatar data has no latents - removing from cache and reprocessing");
                    _avatarAnimationCache.Remove(memoryCacheKey);
                }
                else
                {
                    _avatarData = cachedAvatarData;
                    return cachedAvatarData;
                }
            }
            
            // Check disk cache second (slower but persistent across sessions)
            AvatarData diskCachedData = null;
            if (_diskCache != null && !string.IsNullOrEmpty(diskCacheKey))
            {
                try
                {
                    diskCachedData = await _diskCache.TryLoadAvatarDataAsync(diskCacheKey);
                    if (diskCachedData != null && diskCachedData.Latents.Count > 0)
                    {
                        Logger.Log($"[MuseTalkInference] Using disk cached avatar data for {avatarTextures.Length} textures - {diskCachedData.FaceRegions.Count} faces, {diskCachedData.Latents.Count} latents");
                        
                        // If disk cache only has latents but missing texture data, we need to reprocess face regions
                        bool needsTextureReprocessing = false;
                        if (diskCachedData.FaceRegions.Count > 0)
                        {
                            var firstFace = diskCachedData.FaceRegions[0];
                            if (firstFace.OriginalTexture.data == null || firstFace.CroppedFaceTexture.data == null || 
                                firstFace.BlurredMask.data == null || firstFace.FaceLarge.data == null)
                            {
                                needsTextureReprocessing = true;
                                Logger.Log("[MuseTalkInference] Disk cache has latents but missing texture data, will reprocess face regions for blending");
                            }
                        }
                        
                        if (!needsTextureReprocessing)
                        {
                            // Complete cache hit - store in memory cache for faster future access
                            ManageCacheSize();
                            _avatarAnimationCache[memoryCacheKey] = diskCachedData;
                            
                            _avatarData = diskCachedData;
                            return diskCachedData;
                        }
                        else
                        {
                            // Partial cache hit - we have latents but need to reprocess textures
                            // This will be handled after face detection by using cached latents
                            Logger.Log("[MuseTalkInference] Using cached latents but reprocessing face regions");
                        }
                    }
                }
                catch (Exception e)
                {
                    Logger.LogWarning($"[MuseTalkInference] Failed to load from disk cache: {e.Message}");
                }
            }
            
            Logger.Log($"[MuseTalkInference] Processing new avatar animation sequence with {avatarTextures.Length} textures");
            
            var avatarData = new AvatarData();

            var frames = new List<Frame>();
            foreach (var texture in avatarTextures)
            {
                var frame = TextureUtils.Texture2DToFrame(texture);
                frames.Add(frame);
            }
            
            // Face Detection
            var result = _faceAnalysis.GetLandmarkAndBbox(frames);
            List<Vector4> coordsList = result.Item1;
            List<Frame> framesList = result.Item2;
            
            Logger.Log($"[MuseTalkInference] Face detection completed: {coordsList.Count} results for {avatarTextures.Length} input textures");
            
            // Process each detected face region
            for (int i = 0; i < coordsList.Count; i++)
            {
                var bbox = coordsList[i];
                
                if (bbox == Vector4.zero)
                {
                    Logger.LogWarning($"[MuseTalkInference] No face detected in image {i}, skipping");
                    continue;
                }
                
                try
                {                    
                    var originalTexture = framesList[i];
                    
                    // Crop face region with version-specific margins
                    var croppedTexture = _faceAnalysis.CropFaceRegion(originalTexture, bbox, _config.Version);
                    
                    // Pre-compute segmentation mask and cached data for blending
                    var segmentationData = PrecomputeSegmentationData(originalTexture, bbox, _config.Version);
                   
                    // Create face data for this region using byte arrays
                    var faceData = new FaceData
                    {
                        HasFace = true,
                        BoundingBox = new Rect(bbox.x, bbox.y, bbox.z - bbox.x, bbox.w - bbox.y),
                        CroppedFaceTexture = croppedTexture,
                        OriginalTexture = originalTexture,
                        FaceLarge = segmentationData.FaceLarge,
                        SegmentationMask = segmentationData.SegmentationMask,
                        AdjustedFaceBbox = segmentationData.AdjustedFaceBbox,
                        CropBox = segmentationData.CropBox,
                        
                        MaskSmall = segmentationData.MaskSmall,
                        FullMask = segmentationData.FullMask,
                        BoundaryMask = segmentationData.BoundaryMask,
                        BlurredMask = segmentationData.BlurredMask
                    };
                    
                    avatarData.FaceRegions.Add(faceData);
                    
                    // Get latents for UNet - use cached latents if available
                    float[] latents;
                    if (diskCachedData != null && i < diskCachedData.Latents.Count)
                    {
                        // Use cached latents from disk
                        latents = diskCachedData.Latents[i];
                        Logger.Log($"[MuseTalkInference] Using cached latents for face region {i}");
                    }
                    else
                    {
                        // Generate new latents
                        latents = await GetLatentsForUNet(croppedTexture);
                    }
                    avatarData.Latents.Add(latents);
                }
                catch (Exception e)
                {
                    Logger.LogError($"[MuseTalkInference] Error processing face region {i}: {e.Message}");
                    Logger.LogError($"[MuseTalkInference] Stack trace: {e.StackTrace}");
                    // Continue processing other faces, but this will be caught in validation later
                }
            }
            
            // Validate that we have processed at least one face successfully
            if (avatarData.FaceRegions.Count == 0)
            {
                throw new InvalidOperationException($"No faces detected in any of the {avatarTextures.Length} avatar textures. Please check that the images contain visible faces.");
            }
            
            if (avatarData.Latents.Count == 0)
            {
                throw new InvalidOperationException($"Failed to generate latents for any of the {avatarData.FaceRegions.Count} detected faces. Check VAE encoder initialization and texture format.");
            }
            
            if (avatarData.Latents.Count != avatarData.FaceRegions.Count)
            {
                Logger.LogWarning($"[MuseTalkInference] Latent count ({avatarData.Latents.Count}) does not match face region count ({avatarData.FaceRegions.Count}). Some faces may have failed processing.");
            }
            
            // Store avatar data for later use in blending
            _avatarData = avatarData;
            
            // Cache the processed avatar animation for reuse
            ManageCacheSize();
            _avatarAnimationCache[memoryCacheKey] = avatarData;
            
            // Save to disk cache for persistent storage
            if (_diskCache != null && !string.IsNullOrEmpty(diskCacheKey))
            {
                try
                {
                    await _diskCache.SaveAvatarDataAsync(diskCacheKey, avatarData);
                    Logger.Log($"[MuseTalkInference] Saved avatar data to disk cache: {diskCacheKey}");
                }
                catch (Exception e)
                {
                    Logger.LogWarning($"[MuseTalkInference] Failed to save to disk cache: {e.Message}");
                }
            }
            
            Logger.Log($"[MuseTalkInference] Successfully processed {avatarData.FaceRegions.Count} face regions with {avatarData.Latents.Count} latent sets across {avatarTextures.Length} avatar textures");
            
            return avatarData;
        }
        
        /// <summary>
        /// Process audio clip and extract Whisper features
        /// </summary>
        private async Task<AudioFeatures> ProcessAudio(AudioClip audioClip)
        {
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
                
            // Convert AudioClip to float array
            var audioData = AudioUtils.AudioClipToFloatArray(audioClip);
            
            // Convert stereo to mono if needed
            if (audioClip.channels == 2)
            {
                audioData = AudioUtils.StereoToMono(audioData);
            }
            
            AudioFeatures features;
            
            if (_whisperModel == null || !_whisperModel.IsInitialized)
            {
                throw new InvalidOperationException("Whisper model is not initialized. Real Whisper model is required for proper inference.");
            }
            
            // Use ONNX Whisper model
            features = await ExtractWhisperFeatures(audioData, audioClip.frequency);
            
            return features;
        }
        
        /// <summary>
        /// Extract Whisper features using ONNX WhisperModel
        /// </summary>
        private async Task<AudioFeatures> ExtractWhisperFeatures(float[] audioData, int sampleRate)
        {
            return await Task.Run(() =>
            {
                var features = _whisperModel.ProcessAudio(audioData, sampleRate);
                
                if (features == null)
                {
                    throw new InvalidOperationException("Whisper model failed to process audio. Check model loading and input data.");
                }
                
                return features;
            });
        }
        
        /// <summary>
        /// Encode image using VAE encoder
        /// </summary>
        private async Task<float[]> EncodeImage(Frame image)
        {
            return await Task.Run(() =>
            {
                // Resize image data to 256x256 for VAE encoder
                var resizedData = FrameUtils.ResizeFrame(image, 256, 256, SamplingMode.Bilinear);
                var inputTensor = FrameUtils.FrameToTensor(resizedData, 2.0f / 255.0f, -1.0f);
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("image", inputTensor)
                };
                
                // Run VAE encoder
                using var results = _vaeEncoderSession.Run(inputs);

                var latents = results.First(r => r.Name == "latents").AsTensor<float>();
                var result = latents.ToArray();
                
                return result;
            });
        }
        
        /// <summary>
        /// Encode image with lower half masked using VAE encoder
        /// Matches Python's encode_image_with_half_mask exactly
        /// </summary>
        private async Task<float[]> EncodeImageWithMask(Frame image)
        {
            return await Task.Run(() =>
            {
                // Resize image data to 256x256 for VAE encoder
                var resizedData = FrameUtils.ResizeFrame(image, 256, 256, SamplingMode.Bilinear);
                var inputTensor = FrameUtils.FrameToTensor(resizedData, 2.0f / 255.0f, -1.0f, applyLowerHalfMask: true);
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("image", inputTensor)
                };
                
                // Run VAE encoder
                using var results = _vaeEncoderSession.Run(inputs);
                var latents = results.First(r => r.Name == "latents").AsTensor<float>();
                
                return latents.ToArray();
            });
        }
        
        /// <summary>
        /// Get latents for UNet inference, matching Python's get_latents_for_unet exactly
        /// </summary>
        private async Task<float[]> GetLatentsForUNet(Frame image)
        {
            // Get masked latents (lower half masked)
            var maskedLatents = await EncodeImageWithMask(image);
            
            // Get reference latents (full image)
            var refLatents = await EncodeImage(image);
            
            // Match Python concatenation exactly along axis=1 (channel dimension)
            if (maskedLatents.Length != refLatents.Length)
            {
                throw new InvalidOperationException("Masked and reference latents must have same size");
            }
            
            const int batch = 1;
            const int maskedChannels = 4;
            const int refChannels = 4;
            const int totalChannels = maskedChannels + refChannels;
            const int latentHeight = 32;
            const int latentWidth = 32;
            const int spatialSize = latentHeight * latentWidth;
            
            var combinedLatents = new float[batch * totalChannels * latentHeight * latentWidth];
            
            unsafe
            {
                fixed (float* maskedPtr = maskedLatents)
                fixed (float* refPtr = refLatents)
                fixed (float* combinedPtr = combinedLatents)
                {
                    // Copy channel by channel using fast memory operations
                    for (int b = 0; b < batch; b++)
                    {
                        // Copy masked latents channels (0-3) using Buffer.MemoryCopy
                        for (int c = 0; c < maskedChannels; c++)
                        {
                            int srcOffset = b * maskedChannels * spatialSize + c * spatialSize;
                            int dstOffset = b * totalChannels * spatialSize + c * spatialSize;
                            
                            Buffer.MemoryCopy(
                                maskedPtr + srcOffset,
                                combinedPtr + dstOffset,
                                spatialSize * sizeof(float),
                                spatialSize * sizeof(float)
                            );
                        }
                        
                        // Copy reference latents channels (4-7) using Buffer.MemoryCopy
                        for (int c = 0; c < refChannels; c++)
                        {
                            int srcOffset = b * refChannels * spatialSize + c * spatialSize;
                            int dstOffset = b * totalChannels * spatialSize + (maskedChannels + c) * spatialSize;
                            
                            Buffer.MemoryCopy(
                                refPtr + srcOffset,
                                combinedPtr + dstOffset,
                                spatialSize * sizeof(float),
                                spatialSize * sizeof(float)
                            );
                        }
                    }
                }
            }
            
            return combinedLatents;
        }
        
        /// <summary>
        /// Generate video frames using UNet and VAE decoder (STREAMING)
        /// Similar to LivePortrait - yields frames as they're generated for real-time feedback
        /// </summary>
        private IEnumerator GenerateFramesStreaming(AvatarData avatarData, AudioFeatures audioFeatures, int batchSize, OutputStream stream)
        {
            // Use audio length to determine frame count (like Python implementation)
            int numFrames = audioFeatures.FeatureChunks.Count;
            int numBatches = Mathf.CeilToInt((float)numFrames / batchSize);
            
            if (avatarData.Latents.Count == 0)
            {
                Logger.LogError("[MuseTalkInference] No avatar latents available for frame generation");
                yield break;
            }
            
            Logger.Log($"[MuseTalkInference] Processing {numFrames} frames in {numBatches} batches of {batchSize}");
            
            // Create cycled latent list for smooth animation
            var cycleDLatents = new List<float[]>(avatarData.Latents);
            var reversedLatents = new List<float[]>(avatarData.Latents);
            reversedLatents.Reverse();
            cycleDLatents.AddRange(reversedLatents);
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
            {   
                int startIdx = batchIdx * batchSize;
                int endIdx = Math.Min(startIdx + batchSize, numFrames);
                int actualBatchSize = endIdx - startIdx;
                
                Logger.Log($"[MuseTalkInference] Processing batch {batchIdx + 1}/{numBatches} (frames {startIdx}-{endIdx - 1})");
                
                // Prepare batch data using the frame index to cycle through latents
                var latentBatch = PrepareLatentBatchWithCycling(cycleDLatents, startIdx, actualBatchSize);
                var audioBatch = PrepareAudioBatch(audioFeatures.FeatureChunks, startIdx, actualBatchSize);
                
                // Add positional encoding to audio (async)
                var audioWithPETask = AddPositionalEncoding(audioBatch);
                yield return new WaitUntil(() => audioWithPETask.IsCompleted);
                var audioWithPE = audioWithPETask.Result;
                
                // Run UNet inference (async)
                var predictedLatentsTask = RunUNet(latentBatch, audioWithPE);
                yield return new WaitUntil(() => predictedLatentsTask.IsCompleted);
                var predictedLatents = predictedLatentsTask.Result;
                
                // Decode latents to images (async)
                var batchFramesTask = DecodeLatents(predictedLatents, actualBatchSize, startIdx);
                yield return new WaitUntil(() => batchFramesTask.IsCompleted);
                var batchFrames = batchFramesTask.Result;
                
                // Stream frames to output as they're generated
                foreach (var frame in batchFrames)
                {
                    stream.queue.Enqueue(frame);
                }
                
                // Log batch performance
                Logger.Log($"[MuseTalkInference] Batch {batchIdx} completed ({actualBatchSize} frames) - streamed to output");
                
                // Yield control back to allow processing/rendering
                yield return null;
            }
        }

        /// <summary>
        /// Generate video frames using UNet and VAE decoder (LEGACY)
        /// For backward compatibility - returns all frames at once
        /// </summary>
        private async Task<List<Texture2D>> GenerateFrames(AvatarData avatarData, AudioFeatures audioFeatures, int batchSize)
        {
            var frames = new List<Texture2D>();
            
            // Use audio length to determine frame count (like Python implementation)
            int numFrames = audioFeatures.FeatureChunks.Count;
            int numBatches = Mathf.CeilToInt((float)numFrames / batchSize);
            
            if (avatarData.Latents.Count == 0)
            {
                Logger.LogError("[MuseTalkInference] No avatar latents available for frame generation");
                return frames;
            }
            
            Logger.Log($"[MuseTalkInference] Processing {numFrames} frames in {numBatches} batches of {batchSize}");
            
            // Create cycled latent list for smooth animation
            var cycleDLatents = new List<float[]>(avatarData.Latents);
            var reversedLatents = new List<float[]>(avatarData.Latents);
            reversedLatents.Reverse();
            cycleDLatents.AddRange(reversedLatents);
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
            {   
                int startIdx = batchIdx * batchSize;
                int endIdx = Math.Min(startIdx + batchSize, numFrames);
                int actualBatchSize = endIdx - startIdx;
                
                Logger.Log($"[MuseTalkInference] Processing batch {batchIdx + 1}/{numBatches} (frames {startIdx}-{endIdx - 1})");
                
                // Prepare batch data using the frame index to cycle through latents
                var latentBatch = PrepareLatentBatchWithCycling(cycleDLatents, startIdx, actualBatchSize);
                var audioBatch = PrepareAudioBatch(audioFeatures.FeatureChunks, startIdx, actualBatchSize);
                
                // Add positional encoding to audio
                var audioWithPE = await AddPositionalEncoding(audioBatch);
                
                // Run UNet inference
                var predictedLatents = await RunUNet(latentBatch, audioWithPE);
                
                // Decode latents to images
                var batchFrames = await DecodeLatents(predictedLatents, actualBatchSize, startIdx);
                
                frames.AddRange(batchFrames);
                
                // Log batch performance
                Logger.Log($"[MuseTalkInference] Batch {batchIdx} completed ({actualBatchSize} frames)");
            }
            
            return frames;
        }
        
        /// <summary>
        /// Prepare latent batch with proper cycling for frame-based animation
        /// CRITICAL FIX: Match Python input_latent_list_cycle indexing exactly
        /// </summary>
        private DenseTensor<float> PrepareLatentBatchWithCycling(List<float[]> cycleDLatents, int startIdx, int batchSize)
        {
            // CRITICAL FIX: Match Python tensor preparation exactly
            // Python processes latents as [1, 8, 32, 32] then concatenates for batch
            const int channels = 8, height = 32, width = 32;
            int totalSize = batchSize * channels * height * width;
            var flatBatch = new float[totalSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                // Match Python latent cycling exactly
                int globalFrameIdx = startIdx + i;
                var latentIdx = globalFrameIdx % cycleDLatents.Count;
                var latent = cycleDLatents[latentIdx];
                
                // Verify latent size
                const int expectedLatentSize = 1 * channels * height * width;
                if (latent.Length != expectedLatentSize)
                {
                    throw new InvalidOperationException($"Latent size mismatch: got {latent.Length}, expected {expectedLatentSize}");
                }
                
                // OPTIMIZATION 4: Use unsafe copy for latent batch preparation
                const int singleLatentSize = channels * height * width;
                int batchOffset = i * singleLatentSize;
                
                unsafe
                {
                    fixed (float* src = latent)
                    fixed (float* dst = &flatBatch[batchOffset])
                    {
                        Buffer.MemoryCopy(src, dst, singleLatentSize * sizeof(float), singleLatentSize * sizeof(float));
                    }
                }
            }
            
            // Return tensor with proper batch dimension: [batchSize, 8, 32, 32]
            return new DenseTensor<float>(flatBatch, new[] { batchSize, channels, height, width });
        }
        
        /// <summary>
        /// Prepare audio batch for processing
        /// </summary>
        private DenseTensor<float> PrepareAudioBatch(List<float[]> audioChunks, int startIdx, int batchSize)
        {
            // Audio chunks are [50, 384] and we need [batchSize, 50, 384]
            int timeSteps = 50, features = 384;
            int totalSize = batchSize * timeSteps * features;
            var flatBatch = new float[totalSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                var audioIdx = startIdx + i;
                
                if (audioIdx < audioChunks.Count)
                {
                    var chunk = audioChunks[audioIdx];
                    int batchOffset = i * timeSteps * features;
                    
                    for (int idx = 0; idx < chunk.Length && idx < timeSteps * features; idx++)
                    {
                        flatBatch[batchOffset + idx] = chunk[idx];
                    }
                }
            }
            
            return new DenseTensor<float>(flatBatch, new[] { batchSize, timeSteps, features });
        }
        
        /// <summary>
        /// Add positional encoding to audio features
        /// </summary>
        private async Task<DenseTensor<float>> AddPositionalEncoding(DenseTensor<float> audioBatch)
        {
            return await Task.Run(() =>
            {
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("audio_features", audioBatch)
                };
                
                using var results = _positionalEncodingSession.Run(inputs);
                var output = results.First().AsTensor<float>();
                var result = new DenseTensor<float>(output.ToArray(), output.Dimensions.ToArray());
                
                return result;
            });
        }
        
        /// <summary>
        /// Run UNet inference to predict new latents
        /// CRITICAL FIX: Match Python UNet input processing exactly
        /// OPTIMIZED: Eliminate unnecessary copy by returning ONNX tensor directly with persistent memory
        /// </summary>
        private async Task<Tensor<float>> RunUNet(DenseTensor<float> latentBatch, DenseTensor<float> audioBatch)
        {
            return await Task.Run(() =>
            {
                // Match Python timestep preparation
                int batchSize = (int)latentBatch.Dimensions[0];
                var timesteps = new long[batchSize];
                for (int i = 0; i < batchSize; i++)
                    timesteps[i] = 0L;
                
                var timestepTensor = new DenseTensor<long>(timesteps, new[] { batchSize });
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_latents", latentBatch),
                    NamedOnnxValue.CreateFromTensor("timesteps", timestepTensor),
                    NamedOnnxValue.CreateFromTensor("audio_prompts", audioBatch)
                };
                
                _reusableUNetResult?.Dispose();
                
                // Keep the result alive as a member variable to prevent GC of tensor memory
                _reusableUNetResult = _unetSession.Run(inputs);
                var output = _reusableUNetResult.First().AsTensor<float>();
                
                // OPTIMIZATION: Return the tensor directly without copying via ToArray()
                // The tensor memory is kept alive by _reusableUNetResult member variable
                Tensor<float> result;
                if (output is DenseTensor<float> denseTensor)
                {
                    // Create a new DenseTensor that shares the same Memory<float> buffer (zero copy!)
                    // Memory remains valid because _reusableUNetResult keeps it alive
                    result = new DenseTensor<float>(denseTensor.Buffer, denseTensor.Dimensions.ToArray());
                }
                else
                {
                    // Fallback: only copy if we can't access the buffer directly
                    Logger.Log($"[MuseTalkInference] UNet_ONNX_Inference: Fallback to ToArray()");
                    result = new DenseTensor<float>(output.ToArray(), output.Dimensions.ToArray());
                }
                
                return result;
            });
        }
        
        /// <summary>
        /// Decode latents back to images using VAE decoder and apply seamless blending
        /// OPTIMIZED: Uses reusable batch array and efficient memory operations with zero-copy tensor reuse
        /// </summary>
        private async Task<List<Texture2D>> DecodeLatents(Tensor<float> unetOutputBatch, int batchSize, int globalStartIdx = 0)
        {
            // Run ONNX VAE decoder inference on background thread first            
            var blendedFrames = await Task.Run(() =>
            {
                var tensors = new List<Tensor<float>>();
                Tensor<float> batchImageOutput = null; // Keep reference alive
                
                try
                {
                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("latents", unetOutputBatch)
                    };
                    
                    using var results = _vaeDecoderSession.Run(inputs);

                    var batchImageOutputValue = results.First();
                    batchImageOutput = batchImageOutputValue.AsTensor<float>(); // Keep alive
                    
                    // OPTIMIZATION 1: Pre-calculate dimensions once for entire batch
                    var imageDims = new int[batchImageOutput.Dimensions.Length - 1]; // Remove batch dimension
                    for (int i = 1; i < batchImageOutput.Dimensions.Length; i++)
                    {
                        imageDims[i - 1] = (int)batchImageOutput.Dimensions[i];
                    }
                    
                    int imagesPerBatch = imageDims.Aggregate(1, (a, b) => a * b);
                    int totalElements = imagesPerBatch * batchSize;
                    
                    if (_reusableBatchArray.Length < totalElements)
                    {
                        _reusableBatchArray = new float[totalElements];
                    }
                    
                    if (batchImageOutputValue.Value is DenseTensor<float> denseTensor)
                    {
                        // Use DenseTensor.Buffer for direct access
                        denseTensor.Buffer.Span.CopyTo(_reusableBatchArray.AsSpan(0, totalElements));
                    }
                    else
                    {
                        // Fallback to ToArray() if not DenseTensor
                        var sourceArray = batchImageOutput.ToArray();
                        sourceArray.CopyTo(_reusableBatchArray.AsSpan(0, totalElements));
                    }
                    
                    for (int b = 0; b < batchSize; b++)
                    {
                        int offset = b * imagesPerBatch;
                        
                        // Create Memory<float> that references the reusable array directly (no copying!)
                        var imageMemory = _reusableBatchArray.AsMemory(offset, imagesPerBatch);
                        var singleImageTensor = new DenseTensor<float>(imageMemory, imageDims);
                        tensors.Add(singleImageTensor);
                    }
                }
                catch (Exception e)
                {
                    Logger.LogWarning($"[MuseTalkInference] Batch VAE decoding failed: {e.Message}");
                    throw;
                }
                
                var blendedFrames = new List<Frame>();
                for (int i = 0; i < tensors.Count; i++)
                {
                    var tensor = tensors[i];
                    
                    // Calculate global frame index for proper numbering
                    int globalFrameIdx = globalStartIdx + i;
                    
                    // Step 1: Convert tensor to raw decoded texture
                    var rawDecodedTexture = FrameUtils.TensorToFrame(tensor);
                    
                    // Step 2: Resize to face crop dimensions (matching Python cv2.resize)
                    if (_avatarData != null && _avatarData.FaceRegions.Count > 0)
                    {
                        // Get corresponding face bbox for sizing
                        int avatarIndex = globalFrameIdx % (2 *_avatarData.FaceRegions.Count); // cycled latents
                        if (avatarIndex >= _avatarData.FaceRegions.Count)
                        {
                            avatarIndex = (2 * _avatarData.FaceRegions.Count) - 1 - avatarIndex;
                        }
                        var faceData = _avatarData.FaceRegions[avatarIndex];
                        var bbox = faceData.BoundingBox;
                        
                        int targetWidth = Mathf.RoundToInt(bbox.width);
                        int targetHeight = Mathf.RoundToInt(bbox.height);
                        
                        if (_config.Version == "v15")
                        {
                            targetHeight += 10; // extra_margin
                            // Clamp to original image height
                            targetHeight = Mathf.Min(targetHeight, faceData.OriginalTexture.height - Mathf.RoundToInt(bbox.y));
                        }
                        
                        // Resize decoded frame to face crop size
                        var resizedFrame = FrameUtils.ResizeFrame(rawDecodedTexture, targetWidth, targetHeight);
                        
                        // Convert face bbox to Vector4 format for blending (x1, y1, x2, y2)
                        var faceBbox = new Vector4(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);
                        string blendingMode = "jaw"; // version v15
                        // Use precomputed segmentation data for optimal performance
                        var blendedFrame = ImageBlendingHelper.BlendFaceWithOriginal(
                            faceData.OriginalTexture, 
                            resizedFrame, 
                            faceBbox,
                            faceData.CropBox,
                            faceData.BlurredMask, 
                            faceData.FaceLarge, 
                            blendingMode);
                        
                        blendedFrames.Add(blendedFrame);
                    }
                    else
                    {
                        blendedFrames.Add(rawDecodedTexture);
                    }
                }
                return blendedFrames;
            });
            
            // Convert tensors to textures and apply blending on main thread
            var blendedTextures = new List<Texture2D>();
            foreach (var frame in blendedFrames)
            {
                blendedTextures.Add(TextureUtils.FrameToTexture2D(frame));
            }
            
            return blendedTextures;
        }
        
        /// <summary>
        /// Clear the avatar animation cache (useful for memory management)
        /// REFACTORED: No texture cleanup needed since we use byte arrays now
        /// </summary>
        public static void ClearAvatarAnimationCache()
        {
            // No need to destroy textures since we use byte arrays now
            // The garbage collector will handle the byte arrays automatically
            _avatarAnimationCache.Clear();
            Logger.Log("[MuseTalkInference] Cleared avatar animation cache");
        }
        
        /// <summary>
        /// Clear the disk cache (useful for troubleshooting or storage management)
        /// </summary>
        public async Task ClearDiskCacheAsync()
        {
            if (_diskCache != null)
            {
                await _diskCache.ClearCache();
                Logger.Log("[MuseTalkInference] Cleared disk cache");
            }
        }
        
        /// <summary>
        /// Get disk cache statistics (if disk cache is enabled)
        /// </summary>
        public CacheStatistics GetDiskCacheStatistics()
        {
            return _diskCache?.GetCacheStatistics();
        }
        
        /// <summary>
        /// Check if disk cache is enabled and functional
        /// </summary>
        public bool IsDiskCacheEnabled => _config.EnableDiskCache && _diskCache != null;
        
        /// <summary>
        /// Get cache information for debugging
        /// </summary>
        public string GetCacheInfo()
        {
            var memoryEntries = _avatarAnimationCache.Count;
            var diskStats = GetDiskCacheStatistics();
            
            if (diskStats != null)
            {
                return $"Memory Cache: {memoryEntries} entries | Disk Cache: {diskStats.TotalEntries} entries, {diskStats.TotalSizeMB:F1}MB, {diskStats.HitRate:P1} hit rate";
            }
            else
            {
                return $"Memory Cache: {memoryEntries} entries | Disk Cache: Disabled";
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _unetSession?.Dispose();
                _vaeEncoderSession?.Dispose();
                _vaeDecoderSession?.Dispose();
                _positionalEncodingSession?.Dispose();
                _whisperModel?.Dispose();
                _faceAnalysis?.Dispose();
                _reusableUNetResult?.Dispose();
                _diskCache?.Dispose();
                
                _disposed = true;
                Logger.Log("[MuseTalkInference] Disposed");
            }
            
            GC.SuppressFinalize(this);
        }
        
        ~MuseTalkInference()
        {
            Dispose();
        }
    }
} 