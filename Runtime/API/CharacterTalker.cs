using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace MuseTalk.API
{
    using Core;
    using Utils;

    /// <summary>
    /// Integration class for adding talking head capabilities to characters
    /// Similar to how CharacterVoice adds speech generation
    /// </summary>
    public class CharacterTalker : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        // Avatar and configuration
        public Texture2D AvatarTexture { get; private set; }
        public Texture2D[] AvatarSequence { get; private set; }
        public MuseTalkConfig Config { get; private set; }
        
        // Cached video data for optimization
        private List<Texture2D> _cachedFrames = null;
        private AudioClip _lastAudioClip = null;
        
        // Factory reference
        private readonly MuseTalkFactory _factory;
        private bool _disposed = false;
        
        /// <summary>
        /// Create CharacterTalker with single avatar image
        /// </summary>
        internal CharacterTalker(MuseTalkFactory factory, Texture2D avatarTexture, MuseTalkConfig config = null)
        {
            _factory = factory ?? throw new ArgumentNullException(nameof(factory));
            AvatarTexture = avatarTexture ?? throw new ArgumentNullException(nameof(avatarTexture));
            Config = config ?? new MuseTalkConfig();
            
            Logger.Log($"[CharacterTalker] Created with single avatar texture {avatarTexture.width}x{avatarTexture.height}");
        }
        
        /// <summary>
        /// Create CharacterTalker with avatar image sequence
        /// </summary>
        internal CharacterTalker(MuseTalkFactory factory, Texture2D[] avatarSequence, MuseTalkConfig config = null)
        {
            _factory = factory ?? throw new ArgumentNullException(nameof(factory));
            AvatarSequence = avatarSequence ?? throw new ArgumentNullException(nameof(avatarSequence));
            Config = config ?? new MuseTalkConfig();
            
            if (avatarSequence.Length > 0)
                AvatarTexture = avatarSequence[0]; // Use first as primary
                
            Logger.Log($"[CharacterTalker] Created with avatar sequence of {avatarSequence.Length} images");
        }
        
        /// <summary>
        /// Generate talking head video from speech audio
        /// </summary>
        /// <param name="audioClip">Speech audio to synchronize with</param>
        /// <param name="useCache">Whether to use cached results for repeated audio</param>
        /// <returns>MuseTalkResult containing video frames</returns>
        public async Task<MuseTalkResult> GenerateTalkingVideoAsync(AudioClip audioClip, bool useCache = true)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(CharacterTalker));
                
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
                
            try
            {
                Logger.Log($"[CharacterTalker] Generating talking video for audio: {audioClip.name} ({audioClip.length:F2}s)");
                
                // Check cache if enabled
                if (useCache && _cachedFrames != null && _lastAudioClip == audioClip)
                {
                    Logger.Log("[CharacterTalker] Using cached talking video");
                    return new MuseTalkResult
                    {
                        Success = true,
                        GeneratedFrames = new List<Texture2D>(_cachedFrames),
                        FrameCount = _cachedFrames.Count,
                        ProcessedAvatarCount = 0,
                        AudioFeatureCount = 0,
                        BatchCount = 0
                    };
                }
                
                // Generate new video
                MuseTalkResult result;
                if (AvatarSequence != null && AvatarSequence.Length > 1)
                {
                    // Use avatar sequence for more varied animation
                    result = await _factory.GenerateAsync(AvatarSequence, audioClip, Config.BatchSize);
                }
                else
                {
                    // Use single avatar image
                    result = await _factory.GenerateAsync(AvatarTexture, audioClip, Config.BatchSize);
                }
                
                // Cache results if successful and caching is enabled
                if (useCache && result.Success && result.GeneratedFrames != null)
                {
                    _cachedFrames = new List<Texture2D>(result.GeneratedFrames);
                    _lastAudioClip = audioClip;
                    
                    Logger.Log($"[CharacterTalker] Cached {result.GeneratedFrames.Count} frames");
                }
                
                return result;
            }
            catch (Exception e)
            {
                Logger.LogError($"[CharacterTalker] Exception generating talking video: {e.Message}");
                return new MuseTalkResult
                {
                    Success = false,
                    ErrorMessage = e.Message
                };
            }
        }
        
        /// <summary>
        /// Generate a single talking frame from audio at specific time
        /// </summary>
        /// <param name="audioClip">Audio clip</param>
        /// <param name="timeSeconds">Time position in audio</param>
        /// <returns>Generated frame texture or null if failed</returns>
        public async Task<Texture2D> GenerateFrameAtTimeAsync(AudioClip audioClip, float timeSeconds)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(CharacterTalker));
                
            try
            {
                // For simplicity, generate short audio segment around the time
                float segmentDuration = 0.2f; // 200ms segment
                float startTime = Mathf.Max(0, timeSeconds - segmentDuration * 0.5f);
                float endTime = Mathf.Min(audioClip.length, startTime + segmentDuration);
                
                // Create audio segment (this is a simplified approach)
                // In a real implementation, you'd extract the actual audio segment
                var result = await GenerateTalkingVideoAsync(audioClip, false);
                
                if (result.Success && result.GeneratedFrames.Count > 0)
                {
                    // Return middle frame as approximation
                    int frameIndex = result.GeneratedFrames.Count / 2;
                    return result.GeneratedFrames[frameIndex];
                }
                
                return null;
            }
            catch (Exception e)
            {
                Logger.LogError($"[CharacterTalker] Exception generating frame at time: {e.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Get the last generated video frames
        /// </summary>
        public List<Texture2D> GetLastGeneratedFrames()
        {
            return _cachedFrames != null ? new List<Texture2D>(_cachedFrames) : null;
        }
        
        /// <summary>
        /// Clear cached video data
        /// </summary>
        public void ClearCache()
        {
            if (_cachedFrames != null)
            {
                // Clean up cached textures
                foreach (var frame in _cachedFrames)
                {
                    if (frame != null)
                        UnityEngine.Object.DestroyImmediate(frame);
                }
                _cachedFrames.Clear();
                _cachedFrames = null;
            }
            
            _lastAudioClip = null;
            
            Logger.Log("[CharacterTalker] Cleared cache");
        }
        
        /// <summary>
        /// Update avatar texture (clears cache)
        /// </summary>
        public void UpdateAvatar(Texture2D newAvatarTexture)
        {
            if (newAvatarTexture == null)
                throw new ArgumentNullException(nameof(newAvatarTexture));
                
            AvatarTexture = newAvatarTexture;
            AvatarSequence = null; // Clear sequence when using single texture
            ClearCache(); // Clear cache since avatar changed
            
            Logger.Log($"[CharacterTalker] Updated avatar to {newAvatarTexture.width}x{newAvatarTexture.height}");
        }
        
        /// <summary>
        /// Update avatar sequence (clears cache)
        /// </summary>
        public void UpdateAvatarSequence(Texture2D[] newAvatarSequence)
        {
            if (newAvatarSequence == null || newAvatarSequence.Length == 0)
                throw new ArgumentException("Avatar sequence cannot be null or empty");
                
            AvatarSequence = newAvatarSequence;
            AvatarTexture = newAvatarSequence[0]; // Use first as primary
            ClearCache(); // Clear cache since avatar changed
            
            Logger.Log($"[CharacterTalker] Updated avatar sequence to {newAvatarSequence.Length} images");
        }
        
        /// <summary>
        /// Calculate simple hash for audio content (for caching)
        /// </summary>
        private string CalculateAudioHash(AudioClip audioClip)
        {
            if (audioClip == null)
                return string.Empty;
                
            // Simple hash based on audio properties
            // In a real implementation, you might want to hash actual audio data
            return $"{audioClip.name}_{audioClip.length}_{audioClip.frequency}_{audioClip.samples}";
        }
        
        /// <summary>
        /// Get information about this character talker
        /// </summary>
        public string GetInfo()
        {
            string avatarInfo = AvatarSequence != null 
                ? $"Sequence ({AvatarSequence.Length} images)"
                : $"Single ({AvatarTexture?.width}x{AvatarTexture?.height})";
                
            string cacheInfo = _cachedFrames != null 
                ? $"Cached ({_cachedFrames.Count} frames)"
                : "No cache";
                
            return $"CharacterTalker: Avatar={avatarInfo}, Cache={cacheInfo}";
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                ClearCache();
                // Don't dispose the factory as it might be shared
                
                _disposed = true;
                Logger.Log("[CharacterTalker] Disposed");
            }
            
            GC.SuppressFinalize(this);
        }
        
        ~CharacterTalker()
        {
            Dispose();
        }
    }
} 