using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace MuseTalk.API
{
    using Core;
    using Utils;

    public sealed class LivePortaitStream
    {
        public int TotalExpectedFrames { get; set; }

        public LivePortaitStream(int totalExpectedFrames)
        {
            TotalExpectedFrames = totalExpectedFrames;
        }

        internal readonly ConcurrentQueue<Texture2D> queue = new();
        internal CancellationTokenSource cts = new();

        public bool Finished { get; internal set; }

        /// Non-blocking poll. Returns false if no frame is ready yet.
        public bool TryGetNext(out Texture2D tex) => queue.TryDequeue(out tex);

        /// Yield instruction that waits until the *next* frame exists,
        /// then exposes it through the .Texture property.
        public FrameAwaiter WaitForNext() => new(queue);
    }

    /// Custom yield instruction that delivers one Texture2D.
    public sealed class FrameAwaiter : CustomYieldInstruction
    {
        private readonly ConcurrentQueue<Texture2D> _q;
        public Texture2D Texture { get; private set; }

        public FrameAwaiter(ConcurrentQueue<Texture2D> q) => _q = q;

        public override bool keepWaiting
        {
            get
            {
                if (_q.TryDequeue(out var tex))
                {
                    Texture = tex;
                    return false;          // stop waiting – caller resumes
                }
                return true;               // keep waiting this frame
            }
        }
    }

    /// <summary>
    /// Stream for driving frames input - similar to output stream but for input processing
    /// </summary>
    public sealed class DrivingFramesStream
    {
        public int TotalExpectedFrames { get; set; }
        public bool LoadingFinished { get; internal set; }
        public bool ProcessingFinished { get; internal set; }

        public DrivingFramesStream(int totalExpectedFrames)
        {
            TotalExpectedFrames = totalExpectedFrames;
        }

        internal readonly ConcurrentQueue<Texture2D> loadQueue = new();
        internal CancellationTokenSource cts = new();

        /// Non-blocking poll. Returns false if no frame is ready yet.
        public bool TryGetNext(out Texture2D tex) => loadQueue.TryDequeue(out tex);

        /// Check if frames are available for processing
        public bool HasFramesAvailable => !loadQueue.IsEmpty;

        /// Get current queue count
        public int QueueCount => loadQueue.Count;

        /// Yield instruction that waits until the *next* frame exists,
        /// then exposes it through the .Texture property.
        public FrameAwaiter WaitForNext() => new(loadQueue);

        /// Check if we have more frames to process
        public bool HasMoreFrames => !LoadingFinished || HasFramesAvailable;
    }

    /// <summary>
    /// Unified API that provides both LivePortrait and MuseTalk functionality in a single entry point
    /// 
    /// Supports multiple workflows:
    /// 1. LivePortrait only: Generate animated textures from source image + driving frames
    /// 2. MuseTalk only: Generate lip-synced talking head from avatar + audio
    /// 3. Combined workflow: LivePortrait → MuseTalk for complete talking head generation
    /// 4. Character Talker functionality: High-level character-based talking head generation
    /// 
    /// This replaces both LivePortraitMuseTalkAPI, MuseTalkFactory, and CharacterTalker
    /// </summary>
    public class UnifiedTalkingHeadAPI : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        private LivePortraitInference _livePortrait;
        private MuseTalkInference _museTalk;
        private MuseTalkConfig _config;
        private bool _initialized = false;
        private bool _disposed = false;
        private readonly AvatarController _avatarController;
        
        // Character Talker functionality - cached video data for optimization
        private readonly Dictionary<string, List<Texture2D>> _characterCache = new();
        private readonly Dictionary<string, AudioClip> _lastAudioClips = new();
        
        public bool IsInitialized => _initialized;
        public bool LogTiming 
        { 
            get => MuseTalkInference.LogTiming; 
            set => MuseTalkInference.LogTiming = value; 
        }
        public bool EnableLogging
        {
            get => DebugLogger.EnableLogging;
            set => DebugLogger.EnableLogging = value;
        }
        public bool EnableFileDebug
        {
            get => DebugLogger.EnableFileDebug;
            set => DebugLogger.EnableFileDebug = value;
        }
        
        /// <summary>
        /// Initialize the unified API with configuration
        /// </summary>
        public UnifiedTalkingHeadAPI(MuseTalkConfig config, AvatarController avatarController)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _avatarController = avatarController ?? throw new ArgumentNullException(nameof(avatarController));
            
            try
            {
                Logger.Log("[UnifiedTalkingHeadAPI] Initializing unified workflow...");
                
                // Initialize LivePortrait inference
                _livePortrait = new LivePortraitInference(_config);
                
                // Initialize MuseTalk inference
                _museTalk = new MuseTalkInference(_config);
                
                // Verify both systems are initialized
                if (!_livePortrait.IsInitialized)
                {
                    throw new InvalidOperationException("LivePortrait inference failed to initialize");
                }
                
                if (!_museTalk.IsInitialized)
                {
                    throw new InvalidOperationException("MuseTalk inference failed to initialize");
                }
                
                _initialized = true;
                Logger.Log("[UnifiedTalkingHeadAPI] Successfully initialized unified workflow");
            }
            catch (Exception e)
            {
                Logger.LogError($"[UnifiedTalkingHeadAPI] Failed to initialize: {e.Message}");
                _initialized = false;
            }
        }
        
        #region LivePortrait Methods
        
        /// <summary>
        /// Generate animated textures only using LivePortrait (SYNCHRONOUS) - List<Texture2D> overload
        /// </summary>
        public LivePortaitStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, List<Texture2D> drivingFrames)
        {
            if (!_initialized)
                throw new InvalidOperationException("API not initialized");
                
            if (sourceImage == null || drivingFrames == null)
                throw new ArgumentException("Invalid input: source image and driving frames are required");
                
            Logger.Log($"[UnifiedTalkingHeadAPI] Generating animated textures (SYNC): {drivingFrames.Count} driving frames");
            
            var input = new LivePortraitInput
            {
                SourceImage = sourceImage,
                DrivingFrames = drivingFrames
            };

            var stream = new LivePortaitStream(drivingFrames.Count);
            _avatarController.StartCoroutine(_livePortrait.GenerateAsync(input, stream));
            return stream;
        }

        public LivePortaitStream GenerateAnimatedTexturesAsync(Texture2D sourceImage, string drivingFramesPath, int maxFrames = -1)
        {
            if (!_initialized)
                throw new InvalidOperationException("API not initialized");
                
            if (sourceImage == null || string.IsNullOrEmpty(drivingFramesPath))
                throw new ArgumentException("Invalid input: source image and driving frames path are required");

            // Get frame count first to estimate total frames
            var frameFiles = FileUtils.GetFrameFiles(drivingFramesPath, maxFrames);
            if (frameFiles.Length == 0)
            {
                throw new ArgumentException($"No driving frames found in path: {drivingFramesPath}");
            }

            Logger.Log($"[UnifiedTalkingHeadAPI] Starting pipelined processing: {frameFiles.Length} driving frames");
            
            var stream = new LivePortaitStream(frameFiles.Length);
            _avatarController.StartCoroutine(
                _livePortrait.GenerateAsync(sourceImage, frameFiles, stream, _avatarController));
            return stream;
        }
        
        #endregion
        
        #region MuseTalk Methods
        
        /// <summary>
        /// Generate talking head video from avatar and audio (STREAMING)
        /// </summary>
        /// <param name="avatarTexture">Avatar image texture</param>
        /// <param name="audioClip">Speech audio clip</param>
        /// <param name="batchSize">Processing batch size (default: 4)</param>
        /// <returns>MuseTalkStream for receiving frames as they're generated</returns>
        public MuseTalkStream GenerateMuseTalkStreamingAsync(Texture2D avatarTexture, AudioClip audioClip, int batchSize = 4)
        {
            if (!_initialized)
                throw new InvalidOperationException("API not initialized");
                
            if (avatarTexture == null || audioClip == null)
                throw new ArgumentException("Avatar texture and audio clip are required");
                
            Logger.Log($"[UnifiedTalkingHeadAPI] Starting MuseTalk streaming generation: {audioClip.name} ({audioClip.length:F2}s)");
            
            var input = new Core.MuseTalkInput(avatarTexture, audioClip)
            {
                BatchSize = batchSize
            };
            
            // Estimate frame count based on audio length (approximation)
            int estimatedFrames = Mathf.CeilToInt(audioClip.length * 25f); // ~25 FPS estimate
            var stream = new MuseTalkStream(estimatedFrames);
            
            _avatarController.StartCoroutine(_museTalk.GenerateAsync(input, stream));
            return stream;
        }

        /// <summary>
        /// Generate talking head video from multiple avatar images (STREAMING)
        /// </summary>
        /// <param name="avatarTextures">Array of avatar image textures</param>
        /// <param name="audioClip">Speech audio clip</param>
        /// <param name="batchSize">Processing batch size (default: 4)</param>
        /// <returns>MuseTalkStream for receiving frames as they're generated</returns>
        public MuseTalkStream GenerateMuseTalkStreamingAsync(Texture2D[] avatarTextures, AudioClip audioClip, int batchSize = 4)
        {
            if (!_initialized)
                throw new InvalidOperationException("API not initialized");
                
            if (avatarTextures == null || avatarTextures.Length == 0 || audioClip == null)
                throw new ArgumentException("Avatar textures and audio clip are required");
                
            Logger.Log($"[UnifiedTalkingHeadAPI] Starting MuseTalk streaming generation: {avatarTextures.Length} avatars, {audioClip.name} ({audioClip.length:F2}s)");
            
            var input = new Core.MuseTalkInput(avatarTextures, audioClip)
            {
                BatchSize = batchSize
            };
            
            // Estimate frame count based on audio length (approximation)
            int estimatedFrames = Mathf.CeilToInt(audioClip.length * 25f); // ~25 FPS estimate
            var stream = new MuseTalkStream(estimatedFrames);
            
            _avatarController.StartCoroutine(_museTalk.GenerateAsync(input, stream));
            return stream;
        }

        /// <summary>
        /// Generate talking head video from avatar and audio (LEGACY)
        /// For backward compatibility - returns all frames at once
        /// </summary>
        /// <param name="avatarTexture">Avatar image texture</param>
        /// <param name="audioClip">Speech audio clip</param>
        /// <param name="batchSize">Processing batch size (default: 4)</param>
        /// <returns>MuseTalkResult containing generated frames</returns>
        public async Task<MuseTalkResult> GenerateMuseTalkAsync(Texture2D avatarTexture, AudioClip audioClip, int batchSize = 4)
        {
            if (!_initialized)
            {
                Logger.LogError("[UnifiedTalkingHeadAPI] API is not initialized.");
                return new MuseTalkResult { Success = false, ErrorMessage = "API not initialized" };
            }
            
            if (avatarTexture == null)
            {
                Logger.LogError("[UnifiedTalkingHeadAPI] Avatar texture is required.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Avatar texture is null" };
            }
            
            if (audioClip == null)
            {
                Logger.LogError("[UnifiedTalkingHeadAPI] Audio clip is required.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Audio clip is null" };
            }
            
            try
            {
                var input = new Core.MuseTalkInput(avatarTexture, audioClip)
                {
                    BatchSize = batchSize
                };
                
                return await _museTalk.GenerateAsync(input);
            }
            catch (Exception e)
            {
                Logger.LogError($"[UnifiedTalkingHeadAPI] Exception during generation: {e.Message}\n{e.StackTrace}");
                return new MuseTalkResult 
                { 
                    Success = false, 
                    ErrorMessage = e.Message 
                };
            }
        }
        
        /// <summary>
        /// Generate talking head video from multiple avatar images (LEGACY)
        /// </summary>
        /// <param name="avatarTextures">Array of avatar image textures</param>
        /// <param name="audioClip">Speech audio clip</param>
        /// <param name="batchSize">Processing batch size (default: 4)</param>
        /// <returns>MuseTalkResult containing generated frames</returns>
        public async Task<MuseTalkResult> GenerateMuseTalkAsync(Texture2D[] avatarTextures, AudioClip audioClip, int batchSize = 4)
        {
            if (!_initialized)
            {
                Logger.LogError("[UnifiedTalkingHeadAPI] API is not initialized.");
                return new MuseTalkResult { Success = false, ErrorMessage = "API not initialized" };
            }
            
            if (avatarTextures == null || avatarTextures.Length == 0)
            {
                Logger.LogError("[UnifiedTalkingHeadAPI] Avatar textures array is required and must not be empty.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Avatar textures array is null or empty" };
            }
            
            if (audioClip == null)
            {
                Logger.LogError("[UnifiedTalkingHeadAPI] Audio clip is required.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Audio clip is null" };
            }
            
            try
            {
                var input = new Core.MuseTalkInput(avatarTextures, audioClip)
                {
                    BatchSize = batchSize
                };
                
                return await _museTalk.GenerateAsync(input);
            }
            catch (Exception e)
            {
                Logger.LogError($"[UnifiedTalkingHeadAPI] Exception during generation: {e.Message}\n{e.StackTrace}");
                return new MuseTalkResult 
                { 
                    Success = false, 
                    ErrorMessage = e.Message 
                };
            }
        }

        /// <summary>
        /// Generate talking head video with custom input configuration (LEGACY)
        /// </summary>
        /// <param name="input">Complete MuseTalk input configuration</param>
        /// <returns>MuseTalkResult containing generated frames</returns>
        public async Task<MuseTalkResult> GenerateMuseTalkAsync(Core.MuseTalkInput input)
        {
            if (!_initialized)
            {
                Logger.LogError("[UnifiedTalkingHeadAPI] API is not initialized.");
                return new MuseTalkResult { Success = false, ErrorMessage = "API not initialized" };
            }
            
            if (input == null)
            {
                Logger.LogError("[UnifiedTalkingHeadAPI] Input configuration is required.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Input is null" };
            }
            
            try
            {
                return await _museTalk.GenerateAsync(input);
            }
            catch (Exception e)
            {
                Logger.LogError($"[UnifiedTalkingHeadAPI] Exception during generation: {e.Message}\n{e.StackTrace}");
                return new MuseTalkResult 
                { 
                    Success = false, 
                    ErrorMessage = e.Message 
                };
            }
        }

        /// <summary>
        /// Generate talking head video with custom input configuration (STREAMING)
        /// </summary>
        /// <param name="input">Complete MuseTalk input configuration</param>
        /// <returns>MuseTalkStream for receiving frames as they're generated</returns>
        public MuseTalkStream GenerateMuseTalkStreamingAsync(Core.MuseTalkInput input)
        {
            if (!_initialized)
                throw new InvalidOperationException("API is not initialized.");
                
            if (input == null)
                throw new ArgumentException("Input configuration is required");
                
            Logger.Log($"[UnifiedTalkingHeadAPI] Starting MuseTalk streaming generation: {input.AvatarTextures.Length} avatars, {input.AudioClip.name} ({input.AudioClip.length:F2}s)");
            
            // Estimate frame count based on audio length (approximation)
            int estimatedFrames = Mathf.CeilToInt(input.AudioClip.length * 25f); // ~25 FPS estimate
            var stream = new MuseTalkStream(estimatedFrames);
            
            _avatarController.StartCoroutine(_museTalk.GenerateAsync(input, stream));
            return stream;
        }
        
        #endregion
        
        #region Character Talker Methods
        
        /// <summary>
        /// Generate talking head video from speech audio with character-based caching
        /// (Replaces CharacterTalker functionality)
        /// </summary>
        /// <param name="characterId">Unique identifier for the character (for caching)</param>
        /// <param name="avatarTexture">Character's avatar texture</param>
        /// <param name="audioClip">Speech audio to synchronize with</param>
        /// <param name="useCache">Whether to use cached results for repeated audio</param>
        /// <param name="batchSize">Processing batch size</param>
        /// <returns>MuseTalkResult containing video frames</returns>
        public async Task<MuseTalkResult> GenerateCharacterTalkingVideoAsync(
            string characterId, 
            Texture2D avatarTexture, 
            AudioClip audioClip, 
            bool useCache = true, 
            int batchSize = 4)
        {
            if (!_initialized)
                throw new InvalidOperationException("API not initialized");
                
            if (string.IsNullOrEmpty(characterId))
                throw new ArgumentException("Character ID is required");
                
            if (avatarTexture == null)
                throw new ArgumentNullException(nameof(avatarTexture));
                
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
                
            try
            {
                Logger.Log($"[UnifiedTalkingHeadAPI] Generating character talking video for {characterId}: {audioClip.name} ({audioClip.length:F2}s)");
                
                // Check cache if enabled
                string cacheKey = $"{characterId}_{audioClip.name}";
                if (useCache && _characterCache.ContainsKey(cacheKey) && _lastAudioClips.ContainsKey(cacheKey) && _lastAudioClips[cacheKey] == audioClip)
                {
                    Logger.Log($"[UnifiedTalkingHeadAPI] Using cached talking video for {characterId}");
                    return new MuseTalkResult
                    {
                        Success = true,
                        GeneratedFrames = new List<Texture2D>(_characterCache[cacheKey]),
                        FrameCount = _characterCache[cacheKey].Count,
                        ProcessedAvatarCount = 1,
                        AudioFeatureCount = 0,
                        BatchCount = 0
                    };
                }
                
                // Generate new video
                var result = await GenerateMuseTalkAsync(avatarTexture, audioClip, batchSize);
                
                // Cache results if successful and caching is enabled
                if (useCache && result.Success && result.GeneratedFrames != null)
                {
                    _characterCache[cacheKey] = new List<Texture2D>(result.GeneratedFrames);
                    _lastAudioClips[cacheKey] = audioClip;
                    
                    Logger.Log($"[UnifiedTalkingHeadAPI] Cached {result.GeneratedFrames.Count} frames for {characterId}");
                }
                
                return result;
            }
            catch (Exception e)
            {
                Logger.LogError($"[UnifiedTalkingHeadAPI] Exception generating character talking video: {e.Message}");
                return new MuseTalkResult
                {
                    Success = false,
                    ErrorMessage = e.Message
                };
            }
        }

        /// <summary>
        /// Generate talking head video from speech audio with character sequence and caching
        /// </summary>
        /// <param name="characterId">Unique identifier for the character (for caching)</param>
        /// <param name="avatarSequence">Character's avatar image sequence</param>
        /// <param name="audioClip">Speech audio to synchronize with</param>
        /// <param name="useCache">Whether to use cached results for repeated audio</param>
        /// <param name="batchSize">Processing batch size</param>
        /// <returns>MuseTalkResult containing video frames</returns>
        public async Task<MuseTalkResult> GenerateCharacterTalkingVideoAsync(
            string characterId, 
            Texture2D[] avatarSequence, 
            AudioClip audioClip, 
            bool useCache = true, 
            int batchSize = 4)
        {
            if (!_initialized)
                throw new InvalidOperationException("API not initialized");
                
            if (string.IsNullOrEmpty(characterId))
                throw new ArgumentException("Character ID is required");
                
            if (avatarSequence == null || avatarSequence.Length == 0)
                throw new ArgumentException("Avatar sequence cannot be null or empty");
                
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
                
            try
            {
                Logger.Log($"[UnifiedTalkingHeadAPI] Generating character talking video for {characterId}: {avatarSequence.Length} avatars, {audioClip.name} ({audioClip.length:F2}s)");
                
                // Check cache if enabled
                string cacheKey = $"{characterId}_seq_{audioClip.name}";
                if (useCache && _characterCache.ContainsKey(cacheKey) && _lastAudioClips.ContainsKey(cacheKey) && _lastAudioClips[cacheKey] == audioClip)
                {
                    Logger.Log($"[UnifiedTalkingHeadAPI] Using cached talking video for {characterId}");
                    return new MuseTalkResult
                    {
                        Success = true,
                        GeneratedFrames = new List<Texture2D>(_characterCache[cacheKey]),
                        FrameCount = _characterCache[cacheKey].Count,
                        ProcessedAvatarCount = avatarSequence.Length,
                        AudioFeatureCount = 0,
                        BatchCount = 0
                    };
                }
                
                // Generate new video using avatar sequence for more varied animation
                var result = await GenerateMuseTalkAsync(avatarSequence, audioClip, batchSize);
                
                // Cache results if successful and caching is enabled
                if (useCache && result.Success && result.GeneratedFrames != null)
                {
                    _characterCache[cacheKey] = new List<Texture2D>(result.GeneratedFrames);
                    _lastAudioClips[cacheKey] = audioClip;
                    
                    Logger.Log($"[UnifiedTalkingHeadAPI] Cached {result.GeneratedFrames.Count} frames for {characterId}");
                }
                
                return result;
            }
            catch (Exception e)
            {
                Logger.LogError($"[UnifiedTalkingHeadAPI] Exception generating character talking video: {e.Message}");
                return new MuseTalkResult
                {
                    Success = false,
                    ErrorMessage = e.Message
                };
            }
        }

        /// <summary>
        /// Clear cached video data for a specific character
        /// </summary>
        public void ClearCharacterCache(string characterId)
        {
            if (string.IsNullOrEmpty(characterId))
                return;
                
            var keysToRemove = new List<string>();
            foreach (var key in _characterCache.Keys)
            {
                if (key.StartsWith($"{characterId}_"))
                {
                    keysToRemove.Add(key);
                }
            }
            
            foreach (var key in keysToRemove)
            {
                if (_characterCache.ContainsKey(key))
                {
                    // Clean up cached textures
                    foreach (var frame in _characterCache[key])
                    {
                        if (frame != null)
                            UnityEngine.Object.DestroyImmediate(frame);
                    }
                    _characterCache.Remove(key);
                }
                
                if (_lastAudioClips.ContainsKey(key))
                {
                    _lastAudioClips.Remove(key);
                }
            }
            
            Logger.Log($"[UnifiedTalkingHeadAPI] Cleared cache for character: {characterId}");
        }

        /// <summary>
        /// Clear all character caches
        /// </summary>
        public void ClearAllCharacterCaches()
        {
            foreach (var frames in _characterCache.Values)
            {
                foreach (var frame in frames)
                {
                    if (frame != null)
                        UnityEngine.Object.DestroyImmediate(frame);
                }
            }
            
            _characterCache.Clear();
            _lastAudioClips.Clear();
            
            Logger.Log("[UnifiedTalkingHeadAPI] Cleared all character caches");
        }
        
        #endregion
        
        #region Utility Methods
        
        /// <summary>
        /// Create a video from generated frames and audio
        /// </summary>
        /// <param name="result">MuseTalk generation result</param>
        /// <param name="outputPath">Output video file path</param>
        /// <returns>True if video creation was successful</returns>
        public async Task<bool> CreateVideoAsync(MuseTalkResult result, string outputPath)
        {
            if (result == null || !result.Success || result.GeneratedFrames.Count == 0)
            {
                Logger.LogError("[UnifiedTalkingHeadAPI] Invalid result for video creation.");
                return false;
            }
            
            try
            {                
                // For now, save frames as individual images
                Logger.LogWarning("[UnifiedTalkingHeadAPI] Video encoding not yet implemented. Saving frames as images.");
                
                for (int i = 0; i < result.GeneratedFrames.Count; i++)
                {
                    var frame = result.GeneratedFrames[i];
                    var frameBytes = frame.EncodeToPNG();
                    var framePath = $"{outputPath}_frame_{i:D6}.png";
                    await System.IO.File.WriteAllBytesAsync(framePath, frameBytes);
                }
                
                Logger.Log($"[UnifiedTalkingHeadAPI] Saved {result.GeneratedFrames.Count} frames to {outputPath}_frame_*.png");
                return true;
            }
            catch (Exception e)
            {
                Logger.LogError($"[UnifiedTalkingHeadAPI] Exception creating video: {e.Message}");
                return false;
            }
        }

        /// <summary>
        /// Get cache information for debugging and monitoring
        /// </summary>
        public string GetCacheInfo()
        {
            if (!_initialized) return "API not initialized";
            
            var livePortraitInfo = "LivePortrait: No cache info";
            var museTalkInfo = _museTalk?.GetCacheInfo() ?? "MuseTalk: No cache info";
            var characterInfo = $"Character cache: {_characterCache.Count} entries";
            
            return $"{livePortraitInfo} | {museTalkInfo} | {characterInfo}";
        }
        
        /// <summary>
        /// Clear all caches to free memory
        /// </summary>
        public async Task ClearCachesAsync()
        {
            if (_museTalk != null)
            {
                await _museTalk.ClearDiskCacheAsync();
                MuseTalkInference.ClearAvatarAnimationCache();
            }
            
            ClearAllCharacterCaches();
            
            Logger.Log("[UnifiedTalkingHeadAPI] Cleared all caches");
        }

        /// <summary>
        /// Check if the API is ready for generation
        /// </summary>
        public bool IsReady => _initialized && !_disposed;

        /// <summary>
        /// Check if streaming operations are supported
        /// </summary>
        public bool SupportsStreaming => _initialized && !_disposed && _avatarController != null;

        /// <summary>
        /// Get information about the current API state
        /// </summary>
        public string GetInfo()
        {
            if (!_initialized)
                return "UnifiedTalkingHeadAPI: Not initialized";
                
            return $"UnifiedTalkingHeadAPI: Initialized, LogTiming={LogTiming}, EnableLogging={EnableLogging}, EnableFileDebug={EnableFileDebug}, HasAvatarController={_avatarController != null}, CharacterCaches={_characterCache.Count}";
        }
        
        #endregion
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _livePortrait?.Dispose();
                _museTalk?.Dispose();
                ClearAllCharacterCaches();
                _disposed = true;
                Logger.Log("[UnifiedTalkingHeadAPI] Disposed");
            }
        }
        
        ~UnifiedTalkingHeadAPI()
        {
            Dispose();
        }
    }
    
    /// <summary>
    /// Factory for creating UnifiedTalkingHeadAPI instances
    /// This replaces both LivePortraitMuseTalkFactory and MuseTalkFactoryBuilder
    /// </summary>
    public static class UnifiedTalkingHeadFactory
    {
        /// <summary>
        /// Create an instance of the unified API with default configuration
        /// </summary>
        public static UnifiedTalkingHeadAPI Create(AvatarController avatarController, string modelPath = "MuseTalk")
        {
            var config = new MuseTalkConfig(modelPath);
            return new UnifiedTalkingHeadAPI(config, avatarController);
        }
        
        /// <summary>
        /// Create an instance optimized for performance
        /// </summary>
        public static UnifiedTalkingHeadAPI CreateOptimized(AvatarController avatarController, string modelPath = "MuseTalk")
        {
            var config = MuseTalkConfig.CreateOptimized(modelPath);
            return new UnifiedTalkingHeadAPI(config, avatarController);
        }
        
        /// <summary>
        /// Create an instance optimized for development/debugging
        /// </summary>
        public static UnifiedTalkingHeadAPI CreateForDevelopment(AvatarController avatarController, string modelPath = "MuseTalk")
        {
            var config = MuseTalkConfig.CreateForDevelopment(modelPath);
            return new UnifiedTalkingHeadAPI(config, avatarController);
        }
    }
}
