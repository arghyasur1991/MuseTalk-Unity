using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace MuseTalk.Utils
{
    /// <summary>
    /// Dispatcher for executing operations on Unity's main thread
    /// Essential for Unity API calls like EncodeToPNG() and LoadImage() that must run on the main thread
    /// </summary>
    public class MainThreadDispatcher : MonoBehaviour
    {
        private static MainThreadDispatcher _instance;
        private static readonly Queue<Action> _executionQueue = new();
        
        public static MainThreadDispatcher Instance
        {
            get
            {
                if (_instance == null)
                {
                    // Create a new GameObject with MainThreadDispatcher component
                    var go = new GameObject("MainThreadDispatcher");
                    _instance = go.AddComponent<MainThreadDispatcher>();
                    DontDestroyOnLoad(go);
                }
                return _instance;
            }
        }
        
        void Update()
        {
            lock (_executionQueue)
            {
                while (_executionQueue.Count > 0)
                {
                    _executionQueue.Dequeue().Invoke();
                }
            }
        }
        
        /// <summary>
        /// Execute an action on the main thread
        /// </summary>
        public static void Enqueue(Action action)
        {
            lock (_executionQueue)
            {
                _executionQueue.Enqueue(action);
            }
        }
        
        /// <summary>
        /// Execute a function on the main thread and return the result asynchronously
        /// </summary>
        public static Task<T> EnqueueAsync<T>(Func<T> function)
        {
            var tcs = new TaskCompletionSource<T>();
            
            void WrappedFunction()
            {
                try
                {
                    var result = function();
                    tcs.SetResult(result);
                }
                catch (Exception ex)
                {
                    tcs.SetException(ex);
                }
            }
            
            Enqueue(WrappedFunction);
            return tcs.Task;
        }
        
        /// <summary>
        /// Encode texture to PNG on main thread
        /// </summary>
        public static Task<byte[]> EncodeToPNGAsync(Texture2D texture)
        {
            if (texture == null)
                return Task.FromResult<byte[]>(null);
                
            // If we're already on the main thread, execute directly
            if (IsMainThread)
            {
                try
                {
                    return Task.FromResult(texture.EncodeToPNG());
                }
                catch (Exception ex)
                {
                    return Task.FromException<byte[]>(ex);
                }
            }
            
            return EnqueueAsync(() => texture.EncodeToPNG());
        }

        private static Texture2D LoadImage(string pngData, int width, int height)
        {
            Span<byte> pngDataSpan = new byte[pngData.Length];
            Convert.TryFromBase64String(pngData, pngDataSpan, out _);
            var texture = new Texture2D(width, height);
            ImageConversion.LoadImage(texture, pngDataSpan);
            if (texture.format != TextureFormat.RGB24)
            {
                var convertedTexture = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false)
                {
                    name = texture.name
                };
                convertedTexture.SetPixels(texture.GetPixels());
                convertedTexture.Apply();
                
                // Destroy the original texture and use the converted one
                DestroyImmediate(texture);
                texture = convertedTexture;
            }
            return texture;
        }
        
        /// <summary>
        /// Load PNG data into texture on main thread using RGB24 format
        /// </summary>
        public static Task<Texture2D> LoadImageAsync(string pngData, int width = 2, int height = 2)
        {
            if (string.IsNullOrEmpty(pngData))
                return Task.FromResult<Texture2D>(null);
                
            // If we're already on the main thread, execute directly
            if (IsMainThread)
            {
                try
                {
                    var texture = LoadImage(pngData, width, height);
                    return Task.FromResult(texture);
                }
                catch (Exception ex)
                {
                    return Task.FromException<Texture2D>(ex);
                }
            }
                
            return EnqueueAsync(() =>
            {
                var texture = LoadImage(pngData, width, height);
                return texture;
            });
        }
        
        /// <summary>
        /// Check if we're currently on the main thread
        /// </summary>
        public static bool IsMainThread => System.Threading.Thread.CurrentThread.ManagedThreadId == 1;
        
        void OnDestroy()
        {
            if (_instance == this)
                _instance = null;
        }
    }
} 