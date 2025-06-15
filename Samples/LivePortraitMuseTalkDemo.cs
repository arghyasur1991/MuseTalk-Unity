using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace MuseTalk.Samples
{
    using API;
    using Models;
    
    /// <summary>
    /// Demo showcasing the integrated LivePortrait + MuseTalk workflow
    /// 
    /// Workflow:
    /// 1. User provides a single source image (portrait)
    /// 2. User provides driving frames (expressions/head movements) 
    /// 3. User provides audio for lip sync
    /// 4. LivePortrait generates animated textures from source + driving frames
    /// 5. MuseTalk applies lip sync to the animated textures
    /// 6. Result: Complete talking head animation with expressions and lip sync
    /// </summary>
    public class LivePortraitMuseTalkDemo : MonoBehaviour
    {
        [Header("Input Assets")]
        [SerializeField] private Texture2D sourceImage;
        [SerializeField] private Texture2D[] drivingFrames;
        [SerializeField] private AudioClip audioClip;
        
        [Header("Configuration")]
        [SerializeField] private string modelPath = "MuseTalk";
        [SerializeField] private int batchSize = 4;
        [SerializeField] private bool useComposite = false;
        
        [Header("UI")]
        [SerializeField] private Button generateButton;
        [SerializeField] private Button generateAnimatedOnlyButton;
        [SerializeField] private Button clearCacheButton;
        [SerializeField] private Text statusText;
        [SerializeField] private Text performanceText;
        [SerializeField] private RawImage previewImage;
        [SerializeField] private Slider progressSlider;
        
        [Header("Output")]
        [SerializeField] private RawImage[] resultPreviewImages = new RawImage[4];
        
        private LivePortraitMuseTalkAPI _api;
        private LivePortraitMuseTalkResult _lastResult;
        private int _currentPreviewIndex = 0;
        private Coroutine _previewCoroutine;
        
        void Start()
        {
            SetupUI();
            InitializeAPI();
        }
        
        void SetupUI()
        {
            if (generateButton != null)
                generateButton.onClick.AddListener(() => StartCoroutine(GenerateFullWorkflow()));
                
            if (generateAnimatedOnlyButton != null)
                generateAnimatedOnlyButton.onClick.AddListener(() => StartCoroutine(GenerateAnimatedOnly()));
                
            if (clearCacheButton != null)
                clearCacheButton.onClick.AddListener(() => StartCoroutine(ClearCache()));
                
            UpdateStatus("Ready to initialize...");
        }
        
        void InitializeAPI()
        {
            try
            {
                UpdateStatus("Initializing LivePortrait + MuseTalk...");
                
                // Create optimized configuration
                var config = MuseTalkConfig.CreateOptimized(modelPath);
                
                // Initialize the integrated API
                _api = new LivePortraitMuseTalkAPI(config);
                
                if (_api.IsInitialized)
                {
                    UpdateStatus("✓ LivePortrait + MuseTalk initialized successfully!");
                    SetButtonsEnabled(true);
                }
                else
                {
                    UpdateStatus("✗ Failed to initialize - check model files");
                    SetButtonsEnabled(false);
                }
            }
            catch (System.Exception e)
            {
                UpdateStatus($"✗ Initialization error: {e.Message}");
                SetButtonsEnabled(false);
                Debug.LogError($"LivePortraitMuseTalkDemo initialization failed: {e}");
            }
        }
        
        /// <summary>
        /// Generate complete talking head animation (LivePortrait + MuseTalk)
        /// </summary>
        IEnumerator GenerateFullWorkflow()
        {
            if (!ValidateInputs()) yield break;
            
            UpdateStatus("🎬 Starting full workflow generation...");
            SetButtonsEnabled(false);
            
            try
            {
                // Create input for integrated workflow
                var input = new LivePortraitMuseTalkInput(sourceImage, drivingFrames, audioClip)
                {
                    BatchSize = batchSize,
                    UseComposite = useComposite
                };
                
                // Start the generation process
                var task = _api.GenerateAsync(input);
                
                // Show progress while processing
                yield return StartCoroutine(ShowProgress(task));
                
                var result = task.Result;
                _lastResult = result;
                
                if (result.Success)
                {
                    UpdateStatus($"✓ Generation completed! {result.TalkingHeadFrames.Count} talking head frames generated");
                    UpdatePerformanceInfo(result.Metrics);
                    ShowResults(result.TalkingHeadFrames);
                    StartPreviewAnimation(result.TalkingHeadFrames);
                }
                else
                {
                    UpdateStatus($"✗ Generation failed: {result.ErrorMessage}");
                }
            }
            catch (System.Exception e)
            {
                UpdateStatus($"✗ Error during generation: {e.Message}");
                Debug.LogError($"LivePortraitMuseTalkDemo generation error: {e}");
            }
            finally
            {
                SetButtonsEnabled(true);
                if (progressSlider != null) progressSlider.value = 0;
            }
        }
        
        /// <summary>
        /// Generate only animated textures (LivePortrait only)
        /// </summary>
        IEnumerator GenerateAnimatedOnly()
        {
            if (!ValidateInputsForAnimatedOnly()) yield break;
            
            UpdateStatus("🎨 Generating animated textures only...");
            SetButtonsEnabled(false);
            
            try
            {
                var task = _api.GenerateAnimatedTexturesAsync(sourceImage, drivingFrames, useComposite);
                
                yield return StartCoroutine(ShowProgress(task));
                
                var result = task.Result;
                
                if (result.Success)
                {
                    UpdateStatus($"✓ Animated textures generated! {result.GeneratedFrames.Count} frames");
                    ShowResults(result.GeneratedFrames);
                    StartPreviewAnimation(result.GeneratedFrames);
                }
                else
                {
                    UpdateStatus($"✗ Generation failed: {result.ErrorMessage}");
                }
            }
            catch (System.Exception e)
            {
                UpdateStatus($"✗ Error during generation: {e.Message}");
                Debug.LogError($"LivePortraitMuseTalkDemo animated generation error: {e}");
            }
            finally
            {
                SetButtonsEnabled(true);
                if (progressSlider != null) progressSlider.value = 0;
            }
        }
        
        /// <summary>
        /// Clear all caches
        /// </summary>
        IEnumerator ClearCache()
        {
            UpdateStatus("🧹 Clearing caches...");
            
            var task = _api.ClearCachesAsync();
            yield return new WaitUntil(() => task.IsCompleted);
            
            UpdateStatus("✓ Caches cleared");
        }
        
        /// <summary>
        /// Show progress during async operations
        /// </summary>
        IEnumerator ShowProgress<T>(System.Threading.Tasks.Task<T> task)
        {
            float elapsedTime = 0f;
            
            while (!task.IsCompleted)
            {
                elapsedTime += Time.deltaTime;
                
                // Simulate progress (since we don't have real progress from the task)
                float progress = Mathf.PingPong(elapsedTime * 0.5f, 1f);
                if (progressSlider != null)
                    progressSlider.value = progress;
                
                // Update status with elapsed time
                UpdateStatus($"Processing... ({elapsedTime:F1}s elapsed)");
                
                yield return null;
            }
            
            if (progressSlider != null)
                progressSlider.value = 1f;
        }
        
        /// <summary>
        /// Validate inputs for full workflow
        /// </summary>
        bool ValidateInputs()
        {
            if (sourceImage == null)
            {
                UpdateStatus("✗ Please assign a source image");
                return false;
            }
            
            if (drivingFrames == null || drivingFrames.Length == 0)
            {
                UpdateStatus("✗ Please assign driving frames");
                return false;
            }
            
            if (audioClip == null)
            {
                UpdateStatus("✗ Please assign an audio clip");
                return false;
            }
            
            return true;
        }
        
        /// <summary>
        /// Validate inputs for animated textures only
        /// </summary>
        bool ValidateInputsForAnimatedOnly()
        {
            if (sourceImage == null)
            {
                UpdateStatus("✗ Please assign a source image");
                return false;
            }
            
            if (drivingFrames == null || drivingFrames.Length == 0)
            {
                UpdateStatus("✗ Please assign driving frames");
                return false;
            }
            
            return true;
        }
        
        /// <summary>
        /// Show first few results in preview images
        /// </summary>
        void ShowResults(List<Texture2D> frames)
        {
            for (int i = 0; i < resultPreviewImages.Length && i < frames.Count; i++)
            {
                if (resultPreviewImages[i] != null)
                {
                    resultPreviewImages[i].texture = frames[i];
                }
            }
        }
        
        /// <summary>
        /// Start animated preview of results
        /// </summary>
        void StartPreviewAnimation(List<Texture2D> frames)
        {
            if (_previewCoroutine != null)
                StopCoroutine(_previewCoroutine);
                
            if (previewImage != null && frames.Count > 0)
            {
                _previewCoroutine = StartCoroutine(AnimatePreview(frames));
            }
        }
        
        /// <summary>
        /// Animate preview frames
        /// </summary>
        IEnumerator AnimatePreview(List<Texture2D> frames)
        {
            if (frames.Count == 0) yield break;
            
            int frameIndex = 0;
            
            while (true)
            {
                if (previewImage != null && frameIndex < frames.Count)
                {
                    previewImage.texture = frames[frameIndex];
                }
                
                frameIndex = (frameIndex + 1) % frames.Count;
                yield return new WaitForSeconds(1f / 25f); // 25 FPS
            }
        }
        
        /// <summary>
        /// Update status text
        /// </summary>
        void UpdateStatus(string message)
        {
            if (statusText != null)
                statusText.text = message;
                
            Debug.Log($"[LivePortraitMuseTalkDemo] {message}");
        }
        
        /// <summary>
        /// Update performance information
        /// </summary>
        void UpdatePerformanceInfo(WorkflowMetrics metrics)
        {
            if (performanceText != null && _api != null)
            {
                var perfInfo = _api.GetPerformanceInfo(metrics);
                var cacheInfo = _api.GetCacheInfo();
                performanceText.text = $"Performance: {perfInfo}\nCache: {cacheInfo}";
            }
        }
        
        /// <summary>
        /// Enable/disable UI buttons
        /// </summary>
        void SetButtonsEnabled(bool enabled)
        {
            if (generateButton != null)
                generateButton.interactable = enabled;
                
            if (generateAnimatedOnlyButton != null)
                generateAnimatedOnlyButton.interactable = enabled;
                
            if (clearCacheButton != null)
                clearCacheButton.interactable = enabled;
        }
        
        void OnDestroy()
        {
            if (_previewCoroutine != null)
                StopCoroutine(_previewCoroutine);
                
            _api?.Dispose();
        }
        
        #if UNITY_EDITOR
        /// <summary>
        /// Editor helper for setting up demo assets
        /// </summary>
        [ContextMenu("Setup Demo Assets")]
        void SetupDemoAssets()
        {
            // This could load default demo assets from Resources or StreamingAssets
            Debug.Log("Setup demo assets - implement loading of default source image, driving frames, and audio");
        }
        
        /// <summary>
        /// Save last generated frames to disk (for debugging)
        /// </summary>
        [ContextMenu("Save Last Result")]
        void SaveLastResult()
        {
            if (_lastResult != null && _lastResult.Success)
            {
                var path = UnityEditor.EditorUtility.SaveFolderPanel("Save Generated Frames", "", "");
                if (!string.IsNullOrEmpty(path))
                {
                    for (int i = 0; i < _lastResult.TalkingHeadFrames.Count; i++)
                    {
                        var frame = _lastResult.TalkingHeadFrames[i];
                        var bytes = frame.EncodeToPNG();
                        System.IO.File.WriteAllBytes($"{path}/frame_{i:D4}.png", bytes);
                    }
                    Debug.Log($"Saved {_lastResult.TalkingHeadFrames.Count} frames to {path}");
                }
            }
            else
            {
                Debug.LogWarning("No successful result to save");
            }
        }
        #endif
    }
}

// Additional helper components for the demo

namespace MuseTalk.Samples
{
    /// <summary>
    /// Simple frame rate display for performance monitoring
    /// </summary>
    public class FPSDisplay : MonoBehaviour
    {
        [SerializeField] private Text fpsText;
        private float deltaTime = 0.0f;
        
        void Update()
        {
            deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
            
            if (fpsText != null)
            {
                float fps = 1.0f / deltaTime;
                fpsText.text = $"FPS: {fps:F1}";
            }
        }
    }
    
    /// <summary>
    /// Memory usage display for monitoring during generation
    /// </summary>
    public class MemoryDisplay : MonoBehaviour
    {
        [SerializeField] private Text memoryText;
        
        void Update()
        {
            if (memoryText != null)
            {
                long memory = System.GC.GetTotalMemory(false);
                float memoryMB = memory / (1024f * 1024f);
                memoryText.text = $"Memory: {memoryMB:F1} MB";
            }
        }
    }
}
