using System;
using System.Threading.Tasks;
using UnityEngine;

namespace MuseTalk
{
    using Core;
    using Models;
    using Utils;

    /// <summary>
    /// Factory class for creating MuseTalk instances for talking head generation
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
        private bool _disposed = false;
        private bool _initialized = false;
        
        public static DebugLogger Logger = new();
        
        /// <summary>
        /// Initializes a new instance of the MuseTalkFactory
        /// </summary>
        public MuseTalkFactory()
        {
            // Default initialization - models will be loaded when first needed
        }
        
        /// <summary>
        /// Initialize MuseTalk with specified configuration
        /// </summary>
        public bool Initialize(MuseTalkConfig config = null)
        {
            if (_initialized)
                return true;
                
            try
            {
                config ??= new MuseTalkConfig(); // Use default config if none provided
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
        /// Generate talking head video with custom input configuration
        /// </summary>
        /// <param name="input">Complete MuseTalk input configuration</param>
        /// <returns>MuseTalkResult containing generated frames</returns>
        public async Task<MuseTalkResult> GenerateAsync(MuseTalkInput input)
        {
            if (!_initialized)
            {
                Logger.LogError("[MuseTalkFactory] Factory is not initialized. Call Initialize() first.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Factory not initialized" };
            }
            
            if (input == null)
            {
                Logger.LogError("[MuseTalkFactory] Input configuration is required.");
                return new MuseTalkResult { Success = false, ErrorMessage = "Input is null" };
            }
            
            try
            {
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
        /// Create a video from generated frames and audio
        /// </summary>
        /// <param name="result">MuseTalk generation result</param>
        /// <param name="outputPath">Output video file path</param>
        /// <param name="videoParams">Video encoding parameters</param>
        /// <returns>True if video creation was successful</returns>
        public async Task<bool> CreateVideoAsync(MuseTalkResult result, string outputPath)
        {
            if (result == null || !result.Success || result.GeneratedFrames.Count == 0)
            {
                Logger.LogError("[MuseTalkFactory] Invalid result for video creation.");
                return false;
            }
            
            try
            {                
                // For now, save frames as individual images
                Logger.LogWarning("[MuseTalkFactory] Video encoding not yet implemented. Saving frames as images.");
                
                for (int i = 0; i < result.GeneratedFrames.Count; i++)
                {
                    var frame = result.GeneratedFrames[i];
                    var frameBytes = frame.EncodeToPNG();
                    var framePath = $"{outputPath}_frame_{i:D6}.png";
                    await System.IO.File.WriteAllBytesAsync(framePath, frameBytes);
                }
                
                Logger.Log($"[MuseTalkFactory] Saved {result.GeneratedFrames.Count} frames to {outputPath}_frame_*.png");
                return true;
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkFactory] Exception creating video: {e.Message}");
                return false;
            }
        }
        
        /// <summary>
        /// Get information about the current MuseTalk configuration
        /// </summary>
        public string GetInfo()
        {
            if (!_initialized)
                return "MuseTalkFactory: Not initialized";
                
            return $"MuseTalkFactory: Initialized, LogTiming={LogTiming}, EnableLogging={EnableLogging}, EnableFileDebug={EnableFileDebug}";
        }
        
        /// <summary>
        /// Check if the factory is ready for generation
        /// </summary>
        public bool IsReady => _initialized && !_disposed;
        
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