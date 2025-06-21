using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace LiveTalk.API
{
    using Core;
    using Utils;

    /// <summary>
    /// Result from MuseTalk generation
    /// </summary>
    public class MuseTalkResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; }
        public List<Texture2D> GeneratedFrames { get; set; } = new List<Texture2D>();
        public int FrameCount { get; set; }
        
        // Additional metadata
        public int ProcessedAvatarCount { get; set; }
        public int AudioFeatureCount { get; set; }
        public int BatchCount { get; set; }
    }

    /// <summary>
    /// Factory class for creating MuseTalk instances for talking head generation
    /// Enhanced with streaming capabilities similar to LivePortrait
    /// </summary>
    public class MuseTalkFactory : IDisposable
    {
        public static MuseTalkFactory Instance { get; private set; } = new();

        public bool LogTiming 
        { 
            get => MuseTalkInference.LogTiming; 
            set => MuseTalkInference.LogTiming = value; 
        }

        // Removed LogLevel property - use DebugLogger.EnableLogging instead
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
        
        private MuseTalkInference _museTalk;
        private readonly LiveTalkController _avatarController;
        private bool _disposed = false;
        private bool _initialized = false;
        
        internal static DebugLogger Logger = new();
        
        /// <summary>
        /// Initializes a new instance of the MuseTalkFactory
        /// </summary>
        public MuseTalkFactory()
        {
            // Default initialization - models will be loaded when first needed
        }

        /// <summary>
        /// Initializes a new instance of the MuseTalkFactory with avatar controller for streaming
        /// </summary>
        public MuseTalkFactory(LiveTalkController avatarController)
        {
            _avatarController = avatarController;
        }
        
        /// <summary>
        /// Initialize MuseTalk with specified configuration
        /// </summary>
        public bool Initialize(LiveTalkConfig config = null)
        {
            if (_initialized)
                return true;
                
            try
            {
                config ??= new LiveTalkConfig(); // Use default config if none provided
                _museTalk = new MuseTalkInference(config);
                _initialized = _museTalk.IsInitialized;
                
                if (!_initialized)
                {
                    Logger.LogError("[MuseTalkFactory] Failed to initialize MuseTalk inference.");
                }
                else
                {
                    Logger.Log("[MuseTalkFactory] Successfully initialized");
                }
                
                return _initialized;
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkFactory] Exception during initialization: {e.Message}\n{e.StackTrace}");
                _initialized = false;
                return false;
            }
        }
        
        /// <summary>
        /// Generate talking head video from avatar and audio
        /// </summary>
        /// <param name="avatarTexture">Avatar image texture</param>
        /// <param name="audioClip">Speech audio clip</param>
        /// <param name="batchSize">Processing batch size (default: 4)</param>
        /// <returns>MuseTalkResult containing generated frames</returns>
        public async Task<MuseTalkResult> GenerateAsync(Texture2D avatarTexture, AudioClip audioClip, int batchSize = 4)
        {
            if (!_initialized)
            {
                Logger.LogError("[MuseTalkFactory] Factory is not initialized. Call Initialize() first.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Factory not initialized" };
            }
            
            if (avatarTexture == null)
            {
                Logger.LogError("[MuseTalkFactory] Avatar texture is required.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Avatar texture is null" };
            }
            
            if (audioClip == null)
            {
                Logger.LogError("[MuseTalkFactory] Audio clip is required.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Audio clip is null" };
            }
            
            try
            {
                var input = new MuseTalkInput(avatarTexture, audioClip)
                {
                    BatchSize = batchSize
                };
                
                return await _museTalk.GenerateAsync(input);
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkFactory] Exception during generation: {e.Message}\n{e.StackTrace}");
                return new MuseTalkResult 
                { 
                    Success = false, 
                    ErrorMessage = e.Message 
                };
            }
        }
        
        /// <summary>
        /// Generate talking head video from multiple avatar images and audio
        /// </summary>
        /// <param name="avatarTextures">Array of avatar image textures</param>
        /// <param name="audioClip">Speech audio clip</param>
        /// <param name="batchSize">Processing batch size (default: 4)</param>
        /// <returns>MuseTalkResult containing generated frames</returns>
        public async Task<MuseTalkResult> GenerateAsync(Texture2D[] avatarTextures, AudioClip audioClip, int batchSize = 4)
        {
            if (!_initialized)
            {
                Logger.LogError("[MuseTalkFactory] Factory is not initialized. Call Initialize() first.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Factory not initialized" };
            }
            
            if (avatarTextures == null || avatarTextures.Length == 0)
            {
                Logger.LogError("[MuseTalkFactory] Avatar textures array is required and must not be empty.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Avatar textures array is null or empty" };
            }
            
            if (audioClip == null)
            {
                Logger.LogError("[MuseTalkFactory] Audio clip is required.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Audio clip is null" };
            }
            
            try
            {
                var input = new MuseTalkInput(avatarTextures, audioClip)
                {
                    BatchSize = batchSize
                };
                
                return await _museTalk.GenerateAsync(input);
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkFactory] Exception during generation: {e.Message}\n{e.StackTrace}");
                return new MuseTalkResult 
                { 
                    Success = false, 
                    ErrorMessage = e.Message 
                };
            }
        }
        
        /// <summary>
        /// Generate talking head video with streaming output (NEW STREAMING API)
        /// Similar to LivePortrait's streaming approach - yields frames as they're generated
        /// </summary>
        /// <param name="avatarTexture">Avatar image texture</param>
        /// <param name="audioClip">Speech audio clip</param>
        /// <param name="batchSize">Processing batch size (default: 4)</param>
        /// <returns>OutputStream for receiving frames as they're generated</returns>
        public OutputStream GenerateStreamingAsync(Texture2D avatarTexture, AudioClip audioClip, int batchSize = 4)
        {
            if (!_initialized)
                throw new InvalidOperationException("Factory is not initialized. Call Initialize() first.");
                
            if (_avatarController == null)
                throw new InvalidOperationException("Avatar controller is required for streaming operations. Use constructor with AvatarController parameter.");
                
            if (avatarTexture == null || audioClip == null)
                throw new ArgumentException("Avatar texture and audio clip are required");
                
            Logger.Log($"[MuseTalkFactory] Starting streaming generation: {audioClip.name} ({audioClip.length:F2}s)");
            
            var input = new MuseTalkInput(avatarTexture, audioClip)
            {
                BatchSize = batchSize
            };
            
            // Estimate frame count based on audio length (approximation)
            int estimatedFrames = Mathf.CeilToInt(audioClip.length * 25f); // ~25 FPS estimate
            var stream = new OutputStream(estimatedFrames);
            
            _avatarController.StartCoroutine(_museTalk.GenerateAsync(input, stream));
            return stream;
        }

        /// <summary>
        /// Generate talking head video with multiple avatar images (STREAMING)
        /// </summary>
        /// <param name="avatarTextures">Array of avatar image textures</param>
        /// <param name="audioClip">Speech audio clip</param>
        /// <param name="batchSize">Processing batch size (default: 4)</param>
        /// <returns>OutputStream for receiving frames as they're generated</returns>
        public OutputStream GenerateStreamingAsync(Texture2D[] avatarTextures, AudioClip audioClip, int batchSize = 4)
        {
            if (!_initialized)
                throw new InvalidOperationException("Factory is not initialized. Call Initialize() first.");
                
            if (_avatarController == null)
                throw new InvalidOperationException("Avatar controller is required for streaming operations. Use constructor with AvatarController parameter.");
                
            if (avatarTextures == null || avatarTextures.Length == 0 || audioClip == null)
                throw new ArgumentException("Avatar textures and audio clip are required");
                
            Logger.Log($"[MuseTalkFactory] Starting streaming generation: {avatarTextures.Length} avatars, {audioClip.name} ({audioClip.length:F2}s)");
            
            var input = new MuseTalkInput(avatarTextures, audioClip)
            {
                BatchSize = batchSize
            };
            
            // Estimate frame count based on audio length (approximation)
            int estimatedFrames = Mathf.CeilToInt(audioClip.length * 25f); // ~25 FPS estimate
            var stream = new OutputStream(estimatedFrames);
            
            _avatarController.StartCoroutine(_museTalk.GenerateAsync(input, stream));
            return stream;
        }
        
        /// <summary>
        /// Check if the factory is ready for generation
        /// </summary>
        public bool IsReady => _initialized && !_disposed;

        /// <summary>
        /// Check if streaming operations are supported
        /// </summary>
        public bool SupportsStreaming => _initialized && !_disposed && _avatarController != null;
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _museTalk?.Dispose();
                _museTalk = null;
                _initialized = false;
                _disposed = true;
                
                Logger.Log("[MuseTalkFactory] Disposed");
            }
            
            GC.SuppressFinalize(this);
        }
        
        ~MuseTalkFactory()
        {
            Dispose();
        }
    }
}