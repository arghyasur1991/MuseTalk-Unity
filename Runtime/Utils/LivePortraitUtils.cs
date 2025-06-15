using System;
using UnityEngine;

namespace MuseTalk.Utils
{
    using Models;
    /// <summary>
    /// Utility functions for LivePortrait operations
    /// Adapted from utils_crop.py to match exact functionality
    /// </summary>
    public static class LivePortraitUtils
    {
        /// <summary>
        /// Face align transformation matching Python face_align exactly
        /// </summary>
        public static (Texture2D, Matrix4x4) FaceAlign(Texture2D img, Vector2 center, int outputSize, float scale, float rotation)
        {
            // Calculate transformation matrix (matching Python face_align)
            float rotRad = rotation * Mathf.Deg2Rad;
            float cosRot = Mathf.Cos(rotRad);
            float sinRot = Mathf.Sin(rotRad);
            
            // Scale transformation
            float scaleX = scale;
            float scaleY = scale;
            
            // Translation to center
            float tx = outputSize * 0.5f - center.x * scaleX;
            float ty = outputSize * 0.5f - center.y * scaleY;
            
            // Combined transformation matrix
            var M = Matrix4x4.identity;
            M.m00 = scaleX * cosRot;
            M.m01 = -scaleX * sinRot;
            M.m02 = tx;
            M.m10 = scaleY * sinRot;
            M.m11 = scaleY * cosRot;
            M.m12 = ty;
            
            // Apply transformation to create aligned image
            var alignedTexture = ApplyAffineTransform(img, M, outputSize, outputSize);
            
            return (alignedTexture, M);
        }
        
        /// <summary>
        /// Transform 2D points using transformation matrix
        /// Matching Python trans_points2d exactly
        /// </summary>
        public static Vector2[] TransformPoints2D(Vector2[] points, Matrix4x4 M)
        {
            var transformedPoints = new Vector2[points.Length];
            
            for (int i = 0; i < points.Length; i++)
            {
                Vector3 homogeneous = new Vector3(points[i].x, points[i].y, 1f);
                Vector3 transformed = M * homogeneous;
                transformedPoints[i] = new Vector2(transformed.x, transformed.y);
            }
            
            return transformedPoints;
        }
        
        /// <summary>
        /// Calculate distance to bounding box, matching Python distance2bbox
        /// </summary>
        public static Rect[] Distance2Bbox(Vector2[] anchorCenters, float[] bboxPreds, int strideSize = 1)
        {
            var bboxes = new Rect[anchorCenters.Length];
            
            for (int i = 0; i < anchorCenters.Length; i++)
            {
                float cx = anchorCenters[i].x;
                float cy = anchorCenters[i].y;
                
                int baseIndex = i * 4; // 4 values per bbox prediction
                float left = bboxPreds[baseIndex] * strideSize;
                float top = bboxPreds[baseIndex + 1] * strideSize;
                float right = bboxPreds[baseIndex + 2] * strideSize;
                float bottom = bboxPreds[baseIndex + 3] * strideSize;
                
                float x1 = cx - left;
                float y1 = cy - top;
                float x2 = cx + right;
                float y2 = cy + bottom;
                
                bboxes[i] = new Rect(x1, y1, x2 - x1, y2 - y1);
            }
            
            return bboxes;
        }
        
        /// <summary>
        /// Calculate distance to keypoints, matching Python distance2kps
        /// </summary>
        public static Vector2[][] Distance2Kps(Vector2[] anchorCenters, float[] kpsPreds, int numKeypoints = 5)
        {
            var keypoints = new Vector2[anchorCenters.Length][];
            
            for (int i = 0; i < anchorCenters.Length; i++)
            {
                float cx = anchorCenters[i].x;
                float cy = anchorCenters[i].y;
                
                keypoints[i] = new Vector2[numKeypoints];
                
                for (int k = 0; k < numKeypoints; k++)
                {
                    int baseIndex = i * numKeypoints * 2 + k * 2;
                    float dx = kpsPreds[baseIndex];
                    float dy = kpsPreds[baseIndex + 1];
                    
                    keypoints[i][k] = new Vector2(cx + dx, cy + dy);
                }
            }
            
            return keypoints;
        }
        
        /// <summary>
        /// Non-maximum suppression for bounding boxes
        /// Matching Python nms_boxes functionality
        /// </summary>
        public static int[] NmsBoxes(Rect[] boxes, float[] scores, float nmsThreshold)
        {
            var indices = new System.Collections.Generic.List<int>();
            var areas = new float[boxes.Length];
            
            // Calculate areas
            for (int i = 0; i < boxes.Length; i++)
            {
                areas[i] = boxes[i].width * boxes[i].height;
            }
            
            // Sort by scores (descending)
            var sortedIndices = new int[scores.Length];
            for (int i = 0; i < scores.Length; i++)
                sortedIndices[i] = i;
            
            Array.Sort(sortedIndices, (a, b) => scores[b].CompareTo(scores[a]));
            
            var suppressed = new bool[boxes.Length];
            
            for (int i = 0; i < sortedIndices.Length; i++)
            {
                int idx = sortedIndices[i];
                if (suppressed[idx]) continue;
                
                indices.Add(idx);
                
                // Suppress overlapping boxes
                for (int j = i + 1; j < sortedIndices.Length; j++)
                {
                    int idx2 = sortedIndices[j];
                    if (suppressed[idx2]) continue;
                    
                    float iou = CalculateIOU(boxes[idx], boxes[idx2]);
                    if (iou > nmsThreshold)
                    {
                        suppressed[idx2] = true;
                    }
                }
            }
            
            return indices.ToArray();
        }
        
        /// <summary>
        /// Calculate Intersection over Union (IoU) for two rectangles
        /// </summary>
        private static float CalculateIOU(Rect rect1, Rect rect2)
        {
            float x1 = Mathf.Max(rect1.x, rect2.x);
            float y1 = Mathf.Max(rect1.y, rect2.y);
            float x2 = Mathf.Min(rect1.x + rect1.width, rect2.x + rect2.width);
            float y2 = Mathf.Min(rect1.y + rect1.height, rect2.y + rect2.height);
            
            if (x2 <= x1 || y2 <= y1)
                return 0f;
            
            float intersection = (x2 - x1) * (y2 - y1);
            float area1 = rect1.width * rect1.height;
            float area2 = rect2.width * rect2.height;
            float union = area1 + area2 - intersection;
            
            return union > 0 ? intersection / union : 0f;
        }
        
        /// <summary>
        /// Crop image based on landmarks, matching Python crop_image exactly
        /// </summary>
        public static CropResult CropImage(Texture2D image, Vector2[] landmarks, int dsize = 512, float scale = 2.3f, float vyRatio = -0.125f)
        {
            // Find landmark bounds
            float minX = float.MaxValue, minY = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue;
            
            foreach (var landmark in landmarks)
            {
                minX = Mathf.Min(minX, landmark.x);
                minY = Mathf.Min(minY, landmark.y);
                maxX = Mathf.Max(maxX, landmark.x);
                maxY = Mathf.Max(maxY, landmark.y);
            }
            
            // Calculate center and size
            Vector2 center = new Vector2((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);
            float width = maxX - minX;
            float height = maxY - minY;
            
            // Apply scale and vertical offset
            float cropSize = Mathf.Max(width, height) * scale;
            center.y += height * vyRatio;
            
            // Create crop rectangle
            float x = center.x - cropSize * 0.5f;
            float y = center.y - cropSize * 0.5f;
            
            // Clamp to image bounds
            x = Mathf.Max(0, x);
            y = Mathf.Max(0, y);
            float w = Mathf.Min(image.width - x, cropSize);
            float h = Mathf.Min(image.height - y, cropSize);
            
            var cropRect = new Rect(x, y, w, h);
            
            // Perform crop
            var croppedTexture = TextureUtils.CropTexture(image, cropRect);
            var resizedTexture = TextureUtils.ResizeTexture(croppedTexture, dsize, dsize);
            
            // Calculate transformation matrices
            var M_c2o = CalculateCrop2OriginalMatrix(cropRect, dsize);
            var M_o2c = M_c2o.inverse;
            
            UnityEngine.Object.DestroyImmediate(croppedTexture);
            
            return new CropResult
            {
                ImageCrop = resizedTexture,
                M_c2o = M_c2o,
                M_o2c = M_o2c,
                CropRect = cropRect
            };
        }
        
        /// <summary>
        /// Calculate crop-to-original transformation matrix
        /// </summary>
        private static Matrix4x4 CalculateCrop2OriginalMatrix(Rect cropRect, int dsize)
        {
            var matrix = Matrix4x4.identity;
            
            // Scale from dsize to crop dimensions
            matrix.m00 = cropRect.width / dsize;
            matrix.m11 = cropRect.height / dsize;
            
            // Translation to crop position
            matrix.m03 = cropRect.x;
            matrix.m13 = cropRect.y;
            
            return matrix;
        }
        
        /// <summary>
        /// Apply softmax activation, matching Python softmax exactly
        /// </summary>
        public static float[] Softmax(float[] logits, int axis = 1)
        {
            var result = new float[logits.Length];
            
            if (axis == 1) // Apply softmax along axis 1
            {
                // Find max for numerical stability
                float maxVal = float.MinValue;
                foreach (float logit in logits)
                {
                    maxVal = Mathf.Max(maxVal, logit);
                }
                
                // Calculate exp and sum
                float sum = 0f;
                for (int i = 0; i < logits.Length; i++)
                {
                    result[i] = Mathf.Exp(logits[i] - maxVal);
                    sum += result[i];
                }
                
                // Normalize
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] /= sum;
                }
            }
            
            return result;
        }
        
        /// <summary>
        /// Calculate distance ratio between landmark points
        /// Matching Python calculate_distance_ratio exactly
        /// </summary>
        public static float[] CalculateDistanceRatio(Vector2[] landmarks, int idx1, int idx2, int idx3, int idx4)
        {
            if (landmarks.Length <= Mathf.Max(idx1, idx2, idx3, idx4))
                return new float[] { 1.0f };
                
            float dist1 = Vector2.Distance(landmarks[idx1], landmarks[idx2]);
            float dist2 = Vector2.Distance(landmarks[idx3], landmarks[idx4]);
            
            float ratio = dist2 > 0 ? dist1 / dist2 : 1.0f;
            return new float[] { ratio };
        }
        
        /// <summary>
        /// Get rotation matrix from Euler angles, matching Python get_rotation_matrix
        /// </summary>
        public static float[,] GetRotationMatrix(float[] pitch, float[] yaw, float[] roll)
        {
            // Convert degrees to radians
            float p = pitch[0] * Mathf.Deg2Rad;
            float y = yaw[0] * Mathf.Deg2Rad;
            float r = roll[0] * Mathf.Deg2Rad;
            
            // Calculate rotation matrix components
            float cosP = Mathf.Cos(p), sinP = Mathf.Sin(p);
            float cosY = Mathf.Cos(y), sinY = Mathf.Sin(y);
            float cosR = Mathf.Cos(r), sinR = Mathf.Sin(r);
            
            // Combined rotation matrix (ZYX order)
            var matrix = new float[3, 3];
            matrix[0, 0] = cosY * cosR;
            matrix[0, 1] = cosY * sinR;
            matrix[0, 2] = -sinY;
            matrix[1, 0] = sinP * sinY * cosR - cosP * sinR;
            matrix[1, 1] = sinP * sinY * sinR + cosP * cosR;
            matrix[1, 2] = sinP * cosY;
            matrix[2, 0] = cosP * sinY * cosR + sinP * sinR;
            matrix[2, 1] = cosP * sinY * sinR - sinP * cosR;
            matrix[2, 2] = cosP * cosY;
            
            return matrix;
        }
        
        /// <summary>
        /// Transform keypoint array, matching Python transform_keypoint
        /// </summary>
        public static float[] TransformKeypoint(MotionInfo motionInfo)
        {
            // Get keypoints and reshape to [batch, num_kp, 3]
            var kp = motionInfo.Keypoints;
            int batchSize = 1; // Always 1 for single image processing
            int numKp = kp.Length / 3;
            
            var result = new float[batchSize * numKp * 3];
            
            for (int b = 0; b < batchSize; b++)
            {
                for (int k = 0; k < numKp; k++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        int sourceIndex = k * 3 + c;
                        int targetIndex = b * numKp * 3 + k * 3 + c;
                        
                        if (sourceIndex < kp.Length)
                            result[targetIndex] = kp[sourceIndex];
                    }
                }
            }
            
            return result;
        }
        
        /// <summary>
        /// Prepare paste back mask, matching Python prepare_paste_back
        /// </summary>
        public static Texture2D PreparePasteBack(Texture2D maskTexture, Matrix4x4 M_c2o, Vector2Int targetSize)
        {
            // Transform mask using the crop-to-original matrix
            var transformedMask = ApplyAffineTransform(maskTexture, M_c2o, targetSize.x, targetSize.y);
            return transformedMask;
        }
        
        /// <summary>
        /// Paste back predicted face to original image, matching Python paste_back
        /// </summary>
        public static Texture2D PasteBack(Texture2D predictedFace, Matrix4x4 M_c2o, Texture2D originalImage, Texture2D mask)
        {
            // Resize predicted face to match original crop size
            var resizedPrediction = ApplyAffineTransform(predictedFace, M_c2o, originalImage.width, originalImage.height);
            
            // Blend with original image using mask
            var result = BlendWithMask(originalImage, resizedPrediction, mask);
            
            UnityEngine.Object.DestroyImmediate(resizedPrediction);
            return result;
        }
        
        /// <summary>
        /// Concatenate frames for composite output, matching Python concat_frame
        /// </summary>
        public static Texture2D ConcatFrame(Texture2D drivingFrame, Texture2D cropFrame, Texture2D predictedFrame)
        {
            int width = drivingFrame.width + cropFrame.width + predictedFrame.width;
            int height = Mathf.Max(drivingFrame.height, cropFrame.height, predictedFrame.height);
            
            var compositeTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
            
            // Copy driving frame
            var drivingPixels = drivingFrame.GetPixels();
            compositeTexture.SetPixels(0, 0, drivingFrame.width, drivingFrame.height, drivingPixels);
            
            // Copy crop frame
            var cropPixels = cropFrame.GetPixels();
            compositeTexture.SetPixels(drivingFrame.width, 0, cropFrame.width, cropFrame.height, cropPixels);
            
            // Copy predicted frame
            var predictedPixels = predictedFrame.GetPixels();
            compositeTexture.SetPixels(drivingFrame.width + cropFrame.width, 0, predictedFrame.width, predictedFrame.height, predictedPixels);
            
            compositeTexture.Apply();
            return compositeTexture;
        }
        
        /// <summary>
        /// Apply affine transformation to texture (matching cv2.warpAffine)
        /// </summary>
        private static Texture2D ApplyAffineTransform(Texture2D source, Matrix4x4 transform, int outputWidth, int outputHeight)
        {
            var result = new Texture2D(outputWidth, outputHeight, TextureFormat.RGB24, false);
            var resultPixels = new Color[outputWidth * outputHeight];
            var sourcePixels = source.GetPixels();
            
            // Invert transformation for backward mapping
            var invTransform = transform.inverse;
            
            for (int y = 0; y < outputHeight; y++)
            {
                for (int x = 0; x < outputWidth; x++)
                {
                    // Transform output coordinates to source coordinates
                    Vector3 sourcePos = invTransform.MultiplyPoint3x4(new Vector3(x, y, 0));
                    int srcX = Mathf.RoundToInt(sourcePos.x);
                    int srcY = Mathf.RoundToInt(sourcePos.y);
                    
                    int resultIndex = y * outputWidth + x;
                    
                    // Bilinear interpolation or nearest neighbor
                    if (srcX >= 0 && srcX < source.width && srcY >= 0 && srcY < source.height)
                    {
                        int srcIndex = srcY * source.width + srcX;
                        resultPixels[resultIndex] = sourcePixels[srcIndex];
                    }
                    else
                    {
                        resultPixels[resultIndex] = Color.black; // Fill with black for out-of-bounds
                    }
                }
            }
            
            result.SetPixels(resultPixels);
            result.Apply();
            return result;
        }
        
        /// <summary>
        /// Blend two images using a mask
        /// </summary>
        private static Texture2D BlendWithMask(Texture2D background, Texture2D foreground, Texture2D mask)
        {
            var result = new Texture2D(background.width, background.height, TextureFormat.RGB24, false);
            var bgPixels = background.GetPixels();
            var fgPixels = foreground.GetPixels();
            var maskPixels = mask.GetPixels();
            
            for (int i = 0; i < bgPixels.Length; i++)
            {
                float alpha = maskPixels[i].r; // Use red channel as alpha
                Color blended = Color.Lerp(bgPixels[i], fgPixels[i], alpha);
                bgPixels[i] = blended;
            }
            
            result.SetPixels(bgPixels);
            result.Apply();
            return result;
        }
    }
    
    /// <summary>
    /// Result from crop operation
    /// </summary>
    public class CropResult
    {
        public Texture2D ImageCrop { get; set; }
        public Matrix4x4 M_c2o { get; set; } // Crop to original transformation
        public Matrix4x4 M_o2c { get; set; } // Original to crop transformation
        public Rect CropRect { get; set; }
    }
}
