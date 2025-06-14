using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using Newtonsoft.Json;

namespace MuseTalk.Utils
{
    using Models;

    /// <summary>
    /// Simple serializable rectangle to avoid Unity Rect circular reference issues
    /// </summary>
    [Serializable]
    public class SerializableRect
    {
        public float x { get; set; }
        public float y { get; set; }
        public float width { get; set; }
        public float height { get; set; }
        
        public SerializableRect() { }
        
        public SerializableRect(Rect rect)
        {
            x = rect.x;
            y = rect.y;
            width = rect.width;
            height = rect.height;
        }
        
        public Rect ToRect()
        {
            return new Rect(x, y, width, height);
        }
    }
    
    /// <summary>
    /// Simple serializable Vector2 to avoid Unity Vector2 circular reference issues
    /// </summary>
    [Serializable]
    public class SerializableVector2
    {
        public float x { get; set; }
        public float y { get; set; }
        
        public SerializableVector2() { }
        
        public SerializableVector2(Vector2 vector)
        {
            x = vector.x;
            y = vector.y;
        }
        
        public Vector2 ToVector2()
        {
            return new Vector2(x, y);
        }
    }
    
    /// <summary>
    /// Simple serializable Vector4 to avoid Unity Vector4 circular reference issues
    /// </summary>
    [Serializable]
    public class SerializableVector4
    {
        public float x { get; set; }
        public float y { get; set; }
        public float z { get; set; }
        public float w { get; set; }
        
        public SerializableVector4() { }
        
        public SerializableVector4(Vector4 vector)
        {
            x = vector.x;
            y = vector.y;
            z = vector.z;
            w = vector.w;
        }
        
        public Vector4 ToVector4()
        {
            return new Vector4(x, y, z, w);
        }
    }

    /// <summary>
    /// Serializable version of FaceData for disk storage
    /// Stores texture data as PNG bytes and other metadata
    /// Uses custom serializable types to avoid Unity circular reference issues
    /// </summary>
    [Serializable]
    public class SerializableFaceData
    {
        public bool HasFace { get; set; }
        public SerializableRect BoundingBox { get; set; }
        public SerializableVector2[] Landmarks { get; set; }
        public SerializableVector4 AdjustedFaceBbox { get; set; }
        public SerializableVector4 CropBox { get; set; }
        
        // Texture data stored as PNG bytes (base64 encoded)
        public string CroppedFaceTextureData { get; set; }
        public string OriginalTextureData { get; set; }
        public string FaceLargeData { get; set; }
        public string SegmentationMaskData { get; set; }
        public string MaskSmallData { get; set; }
        public string FullMaskData { get; set; }
        public string BoundaryMaskData { get; set; }
        public string BlurredMaskData { get; set; }
        
        // Texture dimensions for reconstruction
        public int CroppedFaceWidth { get; set; }
        public int CroppedFaceHeight { get; set; }
        public int OriginalWidth { get; set; }
        public int OriginalHeight { get; set; }
        public int FaceLargeWidth { get; set; }
        public int FaceLargeHeight { get; set; }
    }

    /// <summary>
    /// Serializable version of AvatarData for disk storage
    /// </summary>
    [Serializable]
    public class SerializableAvatarData
    {
        public List<SerializableFaceData> FaceRegions { get; set; } = new List<SerializableFaceData>();
        public List<float[]> Latents { get; set; } = new List<float[]>();
        public string Version { get; set; }
        public string CacheKey { get; set; }
        public DateTime CreatedAt { get; set; }
        public long SizeBytes { get; set; }
    }

    /// <summary>
    /// Cache metadata for managing disk cache entries
    /// </summary>
    [Serializable]
    public class CacheMetadata
    {
        public Dictionary<string, List<CacheEntry>> AvatarEntries { get; set; } = new Dictionary<string, List<CacheEntry>>();
        public long TotalSizeBytes { get; set; }
        public int Version { get; set; }
        public DateTime LastCleanup { get; set; }
    }

    /// <summary>
    /// Individual cache entry metadata
    /// </summary>
    [Serializable]
    public class CacheEntry
    {
        public string CacheKey { get; set; }
        public string FilePath { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime LastAccessedAt { get; set; }
        public long SizeBytes { get; set; }
        public int AccessCount { get; set; }
    }

    /// <summary>
    /// Disk cache manager for avatar processing results
    /// Provides persistent storage for face detection and latent generation to avoid re-inference
    /// </summary>
    public class AvatarDiskCache : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        private readonly MuseTalkConfig _config;
        private readonly string _cacheDirectory;
        private readonly string _metadataFilePath;
        private CacheMetadata _metadata;
        private bool _disposed = false;

        private Task _loadMetadataTask;
        
        // File extensions
        private const string CACHE_FILE_EXTENSION = ".avatarcache";
        private const string METADATA_FILE_NAME = "cache_metadata.json";
        
        public AvatarDiskCache(MuseTalkConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            
            // Ensure MainThreadDispatcher is available for texture operations
            _ = MainThreadDispatcher.Instance;
            
            // Determine cache directory
            if (string.IsNullOrEmpty(config.CacheDirectory))
            {
                _cacheDirectory = Path.Combine(Application.persistentDataPath, "MuseTalk", "AvatarCache");
            }
            else
            {
                _cacheDirectory = config.CacheDirectory;
            }
            
            _metadataFilePath = Path.Combine(_cacheDirectory, METADATA_FILE_NAME);
            
            // Initialize cache directory and load metadata
            InitializeCacheDirectory();
            _loadMetadataTask = LoadMetadata();
        }
        
        /// <summary>
        /// Initialize cache directory structure
        /// </summary>
        private void InitializeCacheDirectory()
        {
            try
            {
                if (!Directory.Exists(_cacheDirectory))
                {
                    Directory.CreateDirectory(_cacheDirectory);
                    Logger.Log($"[AvatarDiskCache] Created cache directory: {_cacheDirectory}");
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[AvatarDiskCache] Failed to create cache directory: {e.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Load cache metadata or create new if not exists
        /// </summary>
        private async Task LoadMetadata()
        {
            try
            {
                if (File.Exists(_metadataFilePath))
                {
                    var jsonData = await File.ReadAllTextAsync(_metadataFilePath);
                    _metadata = JsonConvert.DeserializeObject<CacheMetadata>(jsonData);
                    
                    // Check version compatibility
                    if (_metadata.Version != _config.CacheVersionNumber)
                    {
                        Logger.Log($"[AvatarDiskCache] Cache version mismatch ({_metadata.Version} vs {_config.CacheVersionNumber}), clearing cache");
                        await ClearCache();
                        return;
                    }
                    
                    Logger.Log($"[AvatarDiskCache] Loaded cache metadata: {_metadata.AvatarEntries.Count} avatars, {_metadata.TotalSizeBytes / (1024 * 1024)}MB");
                }
                else
                {
                    _metadata = new CacheMetadata
                    {
                        Version = _config.CacheVersionNumber,
                        LastCleanup = DateTime.Now
                    };
                    await SaveMetadata();
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[AvatarDiskCache] Failed to load metadata: {e.Message}");
                _metadata = new CacheMetadata
                {
                    Version = _config.CacheVersionNumber,
                    LastCleanup = DateTime.Now
                };
            }
        }
        
        /// <summary>
        /// Save cache metadata to disk
        /// </summary>
        private async Task SaveMetadata()
        {
            try
            {
                var jsonData = JsonConvert.SerializeObject(_metadata, Formatting.Indented);
                await File.WriteAllTextAsync(_metadataFilePath, jsonData);
            }
            catch (Exception e)
            {
                Logger.LogError($"[AvatarDiskCache] Failed to save metadata: {e.Message}");
            }
        }
        
        /// <summary>
        /// Generate cache key for avatar textures
        /// </summary>
        public string GenerateAvatarCacheKey(Texture2D[] avatarTextures, string version)
        {
            var hashes = new List<string>();
            
            foreach (var texture in avatarTextures)
            {
                var hash = GenerateTextureHash(texture);
                hashes.Add(hash);
            }
            
            // Combine texture hashes with version and config parameters
            var combinedHash = string.Join("-", hashes);
            var configHash = $"{version}_{_config.ExtraMargin}_{_config.UseINT8}";
            
            return $"{combinedHash}_{configHash}".Replace("/", "_").Replace("\\", "_");
        }
        
        /// <summary>
        /// Generate a content-based hash for a texture
        /// </summary>
        private string GenerateTextureHash(Texture2D texture)
        {
            unchecked
            {
                int hash = texture.width.GetHashCode();
                hash = hash * 31 + texture.height.GetHashCode();
                hash = hash * 31 + texture.format.GetHashCode();
                
                // Sample pixels for content-based hashing
                var pixels = texture.GetPixels(0, 0, Math.Min(32, texture.width), Math.Min(32, texture.height));
                for (int i = 0; i < Math.Min(100, pixels.Length); i += 10)
                {
                    hash = hash * 31 + pixels[i].GetHashCode();
                }
                
                return hash.ToString("X8");
            }
        }
        
        /// <summary>
        /// Try to load avatar data from cache
        /// </summary>
        public async Task<AvatarData> TryLoadAvatarDataAsync(string cacheKey)
        {
            if (!_config.EnableDiskCache)
                return null;
            
            try
            {
                await _loadMetadataTask;
                var avatarHash = ExtractAvatarHashFromKey(cacheKey);
                
                if (!_metadata.AvatarEntries.TryGetValue(avatarHash, out var entries))
                    return null;
                
                var entry = entries.FirstOrDefault(e => e.CacheKey == cacheKey);
                if (entry == null)
                    return null;
                
                if (!File.Exists(entry.FilePath))
                {
                    // Remove invalid entry
                    entries.Remove(entry);
                    await SaveMetadata();
                    return null;
                }
                
                // Load cached data
                var jsonData = await File.ReadAllTextAsync(entry.FilePath);
                var serializableData = JsonConvert.DeserializeObject<SerializableAvatarData>(jsonData);
                
                // Convert back to AvatarData
                var avatarData = await DeserializeAvatarDataAsync(serializableData);
                
                // Update access statistics
                entry.LastAccessedAt = DateTime.Now;
                entry.AccessCount++;
                await SaveMetadata();
                
                Logger.Log($"[AvatarDiskCache] Loaded avatar data from cache: {cacheKey}");
                return avatarData;
            }
            catch (Exception e)
            {
                Logger.LogError($"[AvatarDiskCache] Failed to load avatar data: {e.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Save avatar data to cache
        /// </summary>
        public async Task SaveAvatarDataAsync(string cacheKey, AvatarData avatarData)
        {
            if (!_config.EnableDiskCache)
                return;
            
            try
            {
                await _loadMetadataTask;
                var avatarHash = ExtractAvatarHashFromKey(cacheKey);
                var fileName = $"{cacheKey}{CACHE_FILE_EXTENSION}";
                var filePath = Path.Combine(_cacheDirectory, fileName);
                
                // Serialize avatar data
                var serializableData = await SerializeAvatarDataAsync(avatarData, cacheKey);
                var jsonData = JsonConvert.SerializeObject(serializableData, Formatting.Indented);
                
                // Write to disk
                await File.WriteAllTextAsync(filePath, jsonData);
                var fileSize = new FileInfo(filePath).Length;
                
                // Update metadata
                if (!_metadata.AvatarEntries.ContainsKey(avatarHash))
                {
                    _metadata.AvatarEntries[avatarHash] = new List<CacheEntry>();
                }
                
                var entry = new CacheEntry
                {
                    CacheKey = cacheKey,
                    FilePath = filePath,
                    CreatedAt = DateTime.Now,
                    LastAccessedAt = DateTime.Now,
                    SizeBytes = fileSize,
                    AccessCount = 1
                };
                
                _metadata.AvatarEntries[avatarHash].Add(entry);
                _metadata.TotalSizeBytes += fileSize;
                
                // Manage cache size and entry limits
                ManageCacheSize();
                ManageAvatarEntries(avatarHash);
                
                await SaveMetadata();
                
                Logger.Log($"[AvatarDiskCache] Saved avatar data to cache: {cacheKey} ({fileSize / 1024}KB)");
            }
            catch (Exception e)
            {
                Logger.LogError($"[AvatarDiskCache] Failed to save avatar data: {e.Message}");
            }
        }
        
        /// <summary>
        /// Extract avatar hash from cache key (first part before config)
        /// </summary>
        private string ExtractAvatarHashFromKey(string cacheKey)
        {
            var parts = cacheKey.Split('_');
            return parts.Length > 0 ? parts[0] : cacheKey;
        }
        
        /// <summary>
        /// Serialize avatar data for disk storage
        /// Uses MainThreadDispatcher to handle Unity texture operations safely
        /// </summary>
        private async Task<SerializableAvatarData> SerializeAvatarDataAsync(AvatarData avatarData, string cacheKey)
        {
            var serializableData = new SerializableAvatarData
            {
                Version = _config.Version,
                CacheKey = cacheKey,
                CreatedAt = DateTime.Now,
                Latents = new List<float[]>(avatarData.Latents)
            };
            
            if (!_config.CacheLatentsOnly)
            {
                                    // Serialize face regions with texture data
                    foreach (var faceData in avatarData.FaceRegions)
                    {
                        var serializableFace = new SerializableFaceData
                        {
                            HasFace = faceData.HasFace,
                            BoundingBox = new SerializableRect(faceData.BoundingBox),
                            Landmarks = faceData.Landmarks?.Select(v => new SerializableVector2(v)).ToArray(),
                            AdjustedFaceBbox = new SerializableVector4(faceData.AdjustedFaceBbox),
                            CropBox = new SerializableVector4(faceData.CropBox)
                        };
                    
                    // Convert textures to PNG byte data using MainThreadDispatcher
                    if (faceData.CroppedFaceTexture != null)
                    {
                        var pngData = await MainThreadDispatcher.EncodeToPNGAsync(faceData.CroppedFaceTexture);
                        if (pngData != null)
                        {
                            serializableFace.CroppedFaceTextureData = Convert.ToBase64String(pngData);
                            serializableFace.CroppedFaceWidth = faceData.CroppedFaceTexture.width;
                            serializableFace.CroppedFaceHeight = faceData.CroppedFaceTexture.height;
                        }
                    }
                    
                    if (faceData.OriginalTexture != null)
                    {
                        var pngData = await MainThreadDispatcher.EncodeToPNGAsync(faceData.OriginalTexture);
                        if (pngData != null)
                        {
                            serializableFace.OriginalTextureData = Convert.ToBase64String(pngData);
                            serializableFace.OriginalWidth = faceData.OriginalTexture.width;
                            serializableFace.OriginalHeight = faceData.OriginalTexture.height;
                        }
                    }
                    
                    if (faceData.FaceLarge != null)
                    {
                        var pngData = await MainThreadDispatcher.EncodeToPNGAsync(faceData.FaceLarge);
                        if (pngData != null)
                        {
                            serializableFace.FaceLargeData = Convert.ToBase64String(pngData);
                            serializableFace.FaceLargeWidth = faceData.FaceLarge.width;
                            serializableFace.FaceLargeHeight = faceData.FaceLarge.height;
                        }
                    }
                    
                    if (faceData.SegmentationMask != null)
                    {
                        var pngData = await MainThreadDispatcher.EncodeToPNGAsync(faceData.SegmentationMask);
                        if (pngData != null)
                            serializableFace.SegmentationMaskData = Convert.ToBase64String(pngData);
                    }
                    
                    if (faceData.MaskSmall != null)
                    {
                        var pngData = await MainThreadDispatcher.EncodeToPNGAsync(faceData.MaskSmall);
                        if (pngData != null)
                            serializableFace.MaskSmallData = Convert.ToBase64String(pngData);
                    }
                    
                    if (faceData.FullMask != null)
                    {
                        var pngData = await MainThreadDispatcher.EncodeToPNGAsync(faceData.FullMask);
                        if (pngData != null)
                            serializableFace.FullMaskData = Convert.ToBase64String(pngData);
                    }
                    
                    if (faceData.BoundaryMask != null)
                    {
                        var pngData = await MainThreadDispatcher.EncodeToPNGAsync(faceData.BoundaryMask);
                        if (pngData != null)
                            serializableFace.BoundaryMaskData = Convert.ToBase64String(pngData);
                    }
                    
                    if (faceData.BlurredMask != null)
                    {
                        var pngData = await MainThreadDispatcher.EncodeToPNGAsync(faceData.BlurredMask);
                        if (pngData != null)
                            serializableFace.BlurredMaskData = Convert.ToBase64String(pngData);
                    }
                    
                    serializableData.FaceRegions.Add(serializableFace);
                }
            }
            
            return serializableData;
        }
        
        /// <summary>
        /// Deserialize avatar data from disk storage
        /// Uses MainThreadDispatcher to handle Unity texture operations safely
        /// </summary>
        private async Task<AvatarData> DeserializeAvatarDataAsync(SerializableAvatarData serializableData)
        {
            var avatarData = new AvatarData
            {
                Latents = new List<float[]>(serializableData.Latents)
            };
            
            if (!_config.CacheLatentsOnly && serializableData.FaceRegions != null)
            {
                                    // Deserialize face regions with texture data
                    foreach (var serializableFace in serializableData.FaceRegions)
                    {
                        var faceData = new FaceData
                        {
                            HasFace = serializableFace.HasFace,
                            BoundingBox = serializableFace.BoundingBox?.ToRect() ?? default(Rect),
                            Landmarks = serializableFace.Landmarks?.Select(v => v.ToVector2()).ToArray(),
                            AdjustedFaceBbox = serializableFace.AdjustedFaceBbox?.ToVector4() ?? default(Vector4),
                            CropBox = serializableFace.CropBox?.ToVector4() ?? default(Vector4)
                        };
                    
                    // Convert PNG byte data back to textures using MainThreadDispatcher
                    if (!string.IsNullOrEmpty(serializableFace.CroppedFaceTextureData))
                    {
                        var pngData = Convert.FromBase64String(serializableFace.CroppedFaceTextureData);
                        faceData.CroppedFaceTexture = await MainThreadDispatcher.LoadImageAsync(pngData, serializableFace.CroppedFaceWidth, serializableFace.CroppedFaceHeight);
                    }
                    
                    if (!string.IsNullOrEmpty(serializableFace.OriginalTextureData))
                    {
                        var pngData = Convert.FromBase64String(serializableFace.OriginalTextureData);
                        faceData.OriginalTexture = await MainThreadDispatcher.LoadImageAsync(pngData, serializableFace.OriginalWidth, serializableFace.OriginalHeight);
                        if (faceData.OriginalTexture != null)
                        {
                            Logger.Log($"[AvatarDiskCache] OriginalTexture loaded {faceData.OriginalTexture.width}x{faceData.OriginalTexture.height}");
                        }
                    }
                    
                    if (!string.IsNullOrEmpty(serializableFace.FaceLargeData))
                    {
                        var pngData = Convert.FromBase64String(serializableFace.FaceLargeData);
                        faceData.FaceLarge = await MainThreadDispatcher.LoadImageAsync(pngData, serializableFace.FaceLargeWidth, serializableFace.FaceLargeHeight);
                    }
                    
                    if (!string.IsNullOrEmpty(serializableFace.SegmentationMaskData))
                    {
                        var pngData = Convert.FromBase64String(serializableFace.SegmentationMaskData);
                        faceData.SegmentationMask = await MainThreadDispatcher.LoadImageAsync(pngData, 2, 2); // Will be resized by LoadImage
                    }
                    
                    if (!string.IsNullOrEmpty(serializableFace.MaskSmallData))
                    {
                        var pngData = Convert.FromBase64String(serializableFace.MaskSmallData);
                        faceData.MaskSmall = await MainThreadDispatcher.LoadImageAsync(pngData, 2, 2);
                    }
                    
                    if (!string.IsNullOrEmpty(serializableFace.FullMaskData))
                    {
                        var pngData = Convert.FromBase64String(serializableFace.FullMaskData);
                        faceData.FullMask = await MainThreadDispatcher.LoadImageAsync(pngData, 2, 2);
                    }
                    
                    if (!string.IsNullOrEmpty(serializableFace.BoundaryMaskData))
                    {
                        var pngData = Convert.FromBase64String(serializableFace.BoundaryMaskData);
                        faceData.BoundaryMask = await MainThreadDispatcher.LoadImageAsync(pngData, 2, 2);
                    }
                    
                    if (!string.IsNullOrEmpty(serializableFace.BlurredMaskData))
                    {
                        var pngData = Convert.FromBase64String(serializableFace.BlurredMaskData);
                        faceData.BlurredMask = await MainThreadDispatcher.LoadImageAsync(pngData, 2, 2);
                    }
                    
                    avatarData.FaceRegions.Add(faceData);
                }
            }
            
            return avatarData;
        }
        
        /// <summary>
        /// Manage cache size by removing old entries if limit is exceeded
        /// </summary>
        private void ManageCacheSize()
        {
            var maxSizeBytes = _config.MaxCacheSizeMB * 1024 * 1024;
            
            if (_metadata.TotalSizeBytes <= maxSizeBytes)
                return;
            
            Logger.Log($"[AvatarDiskCache] Cache size ({_metadata.TotalSizeBytes / (1024 * 1024)}MB) exceeds limit ({_config.MaxCacheSizeMB}MB), cleaning up...");
            
            // Collect all entries with their access patterns
            var allEntries = _metadata.AvatarEntries.Values
                .SelectMany(entries => entries)
                .OrderBy(e => e.LastAccessedAt) // Remove least recently used first
                .ThenBy(e => e.AccessCount) // Then least frequently used
                .ToList();
            
            // Remove entries until size is under limit
            foreach (var entry in allEntries)
            {
                if (_metadata.TotalSizeBytes <= maxSizeBytes)
                    break;
                
                RemoveCacheEntry(entry);
            }
            
            Logger.Log($"[AvatarDiskCache] Cache cleanup completed, new size: {_metadata.TotalSizeBytes / (1024 * 1024)}MB");
        }
        
        /// <summary>
        /// Manage per-avatar entry count limits
        /// </summary>
        private void ManageAvatarEntries(string avatarHash)
        {
            if (!_metadata.AvatarEntries.TryGetValue(avatarHash, out var entries))
                return;
            
            if (entries.Count <= _config.MaxCacheEntriesPerAvatar)
                return;
            
            // Remove oldest entries for this avatar
            var entriesToRemove = entries
                .OrderBy(e => e.LastAccessedAt)
                .Take(entries.Count - _config.MaxCacheEntriesPerAvatar)
                .ToList();
            
            foreach (var entry in entriesToRemove)
            {
                RemoveCacheEntry(entry);
            }
        }
        
        /// <summary>
        /// Remove a cache entry and its file
        /// </summary>
        private void RemoveCacheEntry(CacheEntry entry)
        {
            try
            {
                if (File.Exists(entry.FilePath))
                {
                    File.Delete(entry.FilePath);
                }
                
                // Remove from metadata
                var avatarHash = ExtractAvatarHashFromKey(entry.CacheKey);
                if (_metadata.AvatarEntries.TryGetValue(avatarHash, out var entries))
                {
                    entries.Remove(entry);
                    if (entries.Count == 0)
                    {
                        _metadata.AvatarEntries.Remove(avatarHash);
                    }
                }
                
                _metadata.TotalSizeBytes -= entry.SizeBytes;
            }
            catch (Exception e)
            {
                Logger.LogError($"[AvatarDiskCache] Failed to remove cache entry: {e.Message}");
            }
        }
        
        /// <summary>
        /// Clear all cache data
        /// </summary>
        public async Task ClearCache()
        {
            try
            {
                if (Directory.Exists(_cacheDirectory))
                {
                    var files = Directory.GetFiles(_cacheDirectory, $"*{CACHE_FILE_EXTENSION}");
                    foreach (var file in files)
                    {
                        File.Delete(file);
                    }
                }
                
                _metadata = new CacheMetadata
                {
                    Version = _config.CacheVersionNumber,
                    LastCleanup = DateTime.Now
                };
                
                await SaveMetadata();
                Logger.Log("[AvatarDiskCache] Cache cleared");
            }
            catch (Exception e)
            {
                Logger.LogError($"[AvatarDiskCache] Failed to clear cache: {e.Message}");
            }
        }
        
        /// <summary>
        /// Get cache statistics
        /// </summary>
        public CacheStatistics GetCacheStatistics()
        {
            var totalEntries = _metadata.AvatarEntries.Values.Sum(entries => entries.Count);
            var totalSizeMB = _metadata.TotalSizeBytes / (1024.0 * 1024.0);
            
            return new CacheStatistics
            {
                TotalAvatars = _metadata.AvatarEntries.Count,
                TotalEntries = totalEntries,
                TotalSizeMB = totalSizeMB,
                HitRate = CalculateHitRate(),
                LastCleanup = _metadata.LastCleanup
            };
        }
        
        /// <summary>
        /// Calculate cache hit rate (simplified)
        /// </summary>
        private double CalculateHitRate()
        {
            var totalAccesses = _metadata.AvatarEntries.Values
                .SelectMany(entries => entries)
                .Sum(e => e.AccessCount);
            
            var totalEntries = _metadata.AvatarEntries.Values.Sum(entries => entries.Count);
            
            return totalEntries > 0 ? (double)totalAccesses / totalEntries : 0.0;
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_loadMetadataTask != null && !_loadMetadataTask.IsCompleted)
                {
                    _loadMetadataTask.Dispose();
                }
                _disposed = true;
            }
        }
    }
    
    /// <summary>
    /// Cache statistics information
    /// </summary>
    public class CacheStatistics
    {
        public int TotalAvatars { get; set; }
        public int TotalEntries { get; set; }
        public double TotalSizeMB { get; set; }
        public double HitRate { get; set; }
        public DateTime LastCleanup { get; set; }
    }
} 