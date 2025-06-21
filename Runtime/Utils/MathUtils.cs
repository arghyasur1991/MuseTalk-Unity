using System;
using UnityEngine;
namespace MuseTalk.Utils
{
    public static class MathUtils
    {        
        /// <summary>
        /// Matrix operations matching Python numpy - EXACT MATCH
        /// </summary>
        public static float[,] MatrixMultiply(float[,] a, float[,] b)
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
        
        public static float[,] TransposeMatrix(float[,] matrix)
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
        public static Vector2[] TransformPts(Vector2[] pts, float[,] M)
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
        /// Invert 3x3 matrix - EXACT MATCH with numpy.linalg.inv
        /// </summary>
        public static float[,] InvertMatrix3x3(float[,] matrix)
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
        
        public static float[] AddArrays(float[] a, float[] b)
        {
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
            return result;
        }
        
        public static float[] SubtractArrays(float[] a, float[] b)
        {
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] - b[i];
            }
            return result;
        }
        
        public static float[] MultiplyArrays(float[] a, float[] b)
        {
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
            return result;
        }
        
        public static float[] DivideArrays(float[] a, float[] b)
        {
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] / b[i];
            }
            return result;
        }
        
        /// <summary>
        /// Python: get_rotation_matrix(pitch, yaw, roll) - EXACT MATCH
        /// </summary>
        public static float[,] GetRotationMatrix(float[] pitch, float[] yaw, float[] roll)
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
        /// Python: cv2.invertAffineTransform(M) - EXACT MATCH
        /// Invert a 2x3 affine transformation matrix
        /// </summary>
        public static float[,] InvertAffineTransform(float[,] M)
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

                
        /// <summary>
        /// Python: trans_points2d() - EXACT MATCH
        /// </summary>
        public static Vector2[] TransPoints2D(Vector2[] pts, Matrix4x4 M)
        {
            var result = new Vector2[pts.Length];
            
            for (int i = 0; i < pts.Length; i++)
            {
                Vector2 pt = pts[i];
                // Python: new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
                // Python: new_pt = np.dot(M, new_pt)
                // Python: new_pts[i] = new_pt[0:2]
                Vector3 newPt = new(pt.x, pt.y, 1.0f);
                Vector3 transformed = M.MultiplyPoint3x4(newPt);
                result[i] = new Vector2(transformed.x, transformed.y);
            }
            
            return result;
        }

        public static Matrix4x4 GetCropTransform(float[,] MInv)
        {
            var Mo2c = Matrix4x4.identity;
            Mo2c.m00 = MInv[0, 0]; Mo2c.m01 = MInv[0, 1]; Mo2c.m03 = MInv[0, 2];
            Mo2c.m10 = MInv[1, 0]; Mo2c.m11 = MInv[1, 1]; Mo2c.m13 = MInv[1, 2];
            Mo2c.m20 = 0f; Mo2c.m21 = 0f; Mo2c.m22 = 1f; Mo2c.m23 = 0f;
            Mo2c.m30 = 0f; Mo2c.m31 = 0f; Mo2c.m32 = 0f; Mo2c.m33 = 1f;
            return Mo2c;
        }
    }
}