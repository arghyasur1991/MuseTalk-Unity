using System;
using System.Collections.Generic;
using UnityEngine;

namespace MuseTalk.Utils
{
    using Models;

    /// <summary>
    /// InsightFace helper that implements the hybrid SCRFD+1k3d68 approach
    /// </summary>
    public class InsightFaceHelper : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        private readonly ScrfdModel _scrfdModel;
        private readonly Landmark68Model _landmarkModel;
        private bool _disposed = false;
        public static readonly Vector4 CoordPlaceholder = new(0.0f, 0.0f, 0.0f, 0.0f);
        public bool IsInitialized { get; private set; }
        
        /// <summary>
        /// Initialize InsightFace models
        /// </summary>
        public InsightFaceHelper()
        {
            try
            {
                Logger.Log("[InsightFaceHelper] Initializing SCRFD and 1k3d68 models...");
                
                _scrfdModel = new ScrfdModel();
                _landmarkModel = new Landmark68Model();
                
                IsInitialized = _scrfdModel.IsInitialized && _landmarkModel.IsInitialized;
                
                if (IsInitialized)
                {
                    Logger.Log("[InsightFaceHelper] Successfully initialized hybrid SCRFD+1k3d68 pipeline");
                }
                else
                {
                    Logger.LogError("[InsightFaceHelper] Failed to initialize one or more models");
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[InsightFaceHelper] Initialization failed: {e.Message}");
                IsInitialized = false;
            }
        }
        
        /// <summary>
        /// Get landmark and bbox using hybrid SCRFD+1k3d68 approach
        /// Matches Python get_landmark_and_bbox_insightface exactly
        /// </summary>
        public (List<Vector4>, List<Texture2D>) GetLandmarkAndBbox(Texture2D[] textures, int bboxShift = 0, string version = "v15", string debugDir = null)
        {
            var coordsList = new List<Vector4>();
            var framesList = new List<Texture2D>(textures);
            
            if (!IsInitialized)
            {
                Logger.LogError("[InsightFaceHelper] Models not initialized");
                Logger.LogError($"[InsightFaceHelper] SCRFD initialized: {_scrfdModel?.IsInitialized ?? false}");
                Logger.LogError($"[InsightFaceHelper] Landmark68 initialized: {_landmarkModel?.IsInitialized ?? false}");
                // Return placeholder coordinates for all frames
                for (int i = 0; i < textures.Length; i++)
                {
                    coordsList.Add(CoordPlaceholder);
                }
                return (coordsList, framesList);
            }
            
            Logger.Log($"[InsightFaceHelper] Processing {textures.Length} images with hybrid SCRFD+1k3d68 approach");
            
            var averageRangeMinus = new List<float>();
            var averageRangePlus = new List<float>();
            
            for (int idx = 0; idx < textures.Length; idx++)
            {
                var texture = textures[idx];
                // Step 1: Detect faces using SCRFD
                var (detections, keypoints) = _scrfdModel.DetectFaces(texture, maxFaces: 1);
                
                if (detections.Length == 0)
                {
                    Logger.LogWarning($"[InsightFaceHelper] No face detected in image {idx} ({texture.width}x{texture.height})");
                    coordsList.Add(CoordPlaceholder);
                    continue;
                }
                
                // Get the best detection
                var detection = detections[0];
                var bbox = detection.BoundingBox;
                var scrfdKps = keypoints.Length > 0 ? keypoints[0] : null;
                
                // Step 2: Extract 1k3d68 landmarks using face-aligned crop
                Vector2[] landmarks68 = null;
                Vector4 finalBbox = CoordPlaceholder;
                
                if (scrfdKps != null && scrfdKps.Length >= 5)
                {
                    // Use SCRFD keypoints for better face alignment
                    landmarks68 = _landmarkModel.GetLandmarks(texture, bbox, scrfdKps);
                    
                    if (landmarks68 != null && landmarks68.Length >= 68)
                    {
                        // Calculate final bbox using hybrid approach
                        finalBbox = CalculateHybridBbox(landmarks68, bbox, scrfdKps, bboxShift);
                        
                        // Calculate range information (matching Python)
                        var (rangeMinus, rangePlus) = CalculateLandmarkRanges(landmarks68);
                        averageRangeMinus.Add(rangeMinus);
                        averageRangePlus.Add(rangePlus);
                    }
                    else
                    {
                        Logger.LogWarning($"[InsightFaceHelper] Failed to extract 1k3d68 landmarks for image {idx}");
                        finalBbox = CreateFallbackBbox(bbox, scrfdKps);
                    }
                }
                else
                {
                    Logger.LogWarning($"[InsightFaceHelper] No SCRFD keypoints for image {idx}, using detection bbox");
                    finalBbox = CreateFallbackBbox(bbox, null);
                }

                float width = finalBbox.z - finalBbox.x;
                float height = finalBbox.w - finalBbox.y;
                if (height <= 0 || width <= 0 || finalBbox.x < 0)
                {
                    Logger.LogWarning($"[InsightFaceHelper] Invalid landmark bbox: [{finalBbox.x:F1}, {finalBbox.y:F1}, {finalBbox.z:F1}, {finalBbox.w:F1}], using SCRFD bbox");
                    coordsList.Add(CreateFallbackBbox(bbox, scrfdKps));
                }
                else
                {
                    coordsList.Add(finalBbox);
                }
                
                // Store the processed data
                framesList.Add(texture);
            }
            
            return (coordsList, framesList);
        }
        
        /// <summary>
        /// Calculate hybrid bbox using landmark center + SCRFD-like dimensions
        /// Matches Python hybrid approach exactly
        /// </summary>
        private Vector4 CalculateHybridBbox(Vector2[] landmarks68, Rect originalBbox, Vector2[] scrfdKps, int bboxShift)
        {
            // MATCH PYTHON EXACTLY: Get landmark center and bounds
            // landmark_center_x = np.mean(face_land_mark[:, 0])
            // landmark_center_y = np.mean(face_land_mark[:, 1])
            Vector2 landmarkCenter = Vector2.zero;
            for (int i = 0; i < landmarks68.Length; i++)
            {
                landmarkCenter += landmarks68[i];
            }
            landmarkCenter /= landmarks68.Length;
            
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
            
            // Apply bbox shift if specified (matching Python)
            if (bboxShift != 0)
            {
                // Use landmark 29 like Python for shift reference
                if (landmarks68.Length > 29)
                {
                    Vector2 halfFaceCoord = landmarks68[29];
                    float shiftedY = halfFaceCoord.y + bboxShift;
                    float yOffset = shiftedY - halfFaceCoord.y;
                    y1 += yOffset;
                    y2 += yOffset;
                }
            }
            
            return new Vector4(x1, y1, x2, y2);
        }
        
        /// <summary>
        /// Calculate landmark range information (matching Python)
        /// </summary>
        private (float rangeMinus, float rangePlus) CalculateLandmarkRanges(Vector2[] landmarks68)
        {
            if (landmarks68.Length < 31)
                return (20f, 20f); // Default values
            
            // Use nose area landmarks for range calculation (matching Python)
            float rangeMinus = Mathf.Abs(landmarks68[30].y - landmarks68[29].y);
            float rangePlus = Mathf.Abs(landmarks68[29].y - landmarks68[28].y);
            
            return (rangeMinus, rangePlus);
        }
        
        /// <summary>
        /// Create fallback bbox when landmarks fail (matching Python exactly)
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
        /// Crop face region with version-specific margins (matching Python)
        /// </summary>
        public Texture2D CropFaceRegion(Texture2D originalTexture, Vector4 bbox, string version)
        {
            int x1 = Mathf.RoundToInt(bbox.x);
            int y1 = Mathf.RoundToInt(bbox.y);
            int x2 = Mathf.RoundToInt(bbox.z);
            int y2 = Mathf.RoundToInt(bbox.w);
            
            // Add version-specific margin (matching Python)
            if (version == "v15")
            {
                y2 += 10; // extra margin for v15
                y2 = Mathf.Min(y2, originalTexture.height);
            }
            
            int width = x2 - x1;
            int height = y2 - y1;
            
            if (width <= 0 || height <= 0)
            {
                Logger.LogError($"[InsightFaceHelper] Invalid crop dimensions: {width}x{height}");
                return TextureUtils.ResizeTexture(originalTexture, 256, 256); // Fallback
            }
            
            // Extract face region
            var pixels = originalTexture.GetPixels(x1, originalTexture.height - y2, width, height);
            var croppedTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
            croppedTexture.SetPixels(pixels);
            croppedTexture.Apply();
            
            // Resize to standard size (256x256 for MuseTalk)
            var resizedTexture = TextureUtils.ResizeTexture(croppedTexture, 256, 256);
            UnityEngine.Object.DestroyImmediate(croppedTexture);
            
            return resizedTexture;
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                _scrfdModel?.Dispose();
                _landmarkModel?.Dispose();
                _disposed = true;
                Logger.Log("[InsightFaceHelper] Disposed");
            }
        }
    }
} 