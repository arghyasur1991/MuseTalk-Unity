using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using MuseTalk.API;
using MuseTalk.Core;

/// <summary>
/// Demo showcasing MuseTalk talking head generation
/// Now using the unified API that consolidates all functionality
/// </summary>
public class MuseTalkDemo : MonoBehaviour
{
    [Header("Input")]
    public Texture2D avatarTexture;
    public AudioClip audioClip;
    
    [Header("Configuration")]
    public string modelPath = "MuseTalk";
    public string version = "v15";
    public int batchSize = 4;
    public bool useInt8 = true;
    
    [Header("UI")]
    public Button generateButton;
    public Button clearCacheButton;
    public Text statusText;
    public RawImage outputImage;
    
    [Header("Output")]
    public string outputPath = "MuseTalkOutput";
    public bool saveFrames = true;
    
    private UnifiedTalkingHeadAPI _api;
    private AvatarController _avatarController;
    private Coroutine _playbackCoroutine;
    private bool _initialized = false;
    
    void Start()
    {
        // Setup UI
        if (generateButton) generateButton.onClick.AddListener(OnGenerateClicked);
        if (clearCacheButton) clearCacheButton.onClick.AddListener(OnClearCacheClicked);
        
        UpdateStatus("Initializing MuseTalk...");
        
        // Initialize the API
        StartCoroutine(InitializeAPI());
    }
    
    IEnumerator InitializeAPI()
    {
        try
        {
            // Get or create avatar controller
            _avatarController = FindObjectOfType<AvatarController>();
            if (_avatarController == null)
            {
                var go = new GameObject("AvatarController");
                _avatarController = go.AddComponent<AvatarController>();
            }
            
            // Create the unified API
            _api = UnifiedTalkingHeadFactory.Create(_avatarController, modelPath);
            
            // Wait a frame to let initialization complete
            yield return null;
            
            if (_api.IsInitialized)
            {
                _initialized = true;
                UpdateStatus($"MuseTalk initialized successfully\n{_api.GetInfo()}");
                
                if (generateButton) generateButton.interactable = true;
                if (clearCacheButton) clearCacheButton.interactable = true;
            }
            else
            {
                UpdateStatus("Failed to initialize MuseTalk");
            }
        }
        catch (Exception e)
        {
            UpdateStatus($"Error initializing MuseTalk: {e.Message}");
            Debug.LogError($"MuseTalk initialization error: {e}");
        }
    }
    
    async void OnGenerateClicked()
    {
        if (!_initialized || _api == null)
        {
            UpdateStatus("API not initialized");
            return;
        }
        
        if (avatarTexture == null)
        {
            UpdateStatus("Please assign an avatar texture");
            return;
        }
        
        if (audioClip == null)
        {
            UpdateStatus("Please assign an audio clip");
            return;
        }
        
        try
        {
            UpdateStatus($"Generating talking head...\nAvatar: {avatarTexture.name}\nAudio: {audioClip.name} ({audioClip.length:F2}s)");
            
            if (generateButton) generateButton.interactable = false;
            
            // Generate talking head video using the unified API
            var result = await _api.GenerateMuseTalkAsync(avatarTexture, audioClip, batchSize);
            
            if (result.Success)
            {
                UpdateStatus($"Generated {result.FrameCount} frames successfully!\nProcessed {result.ProcessedAvatarCount} avatars");
                
                // Save frames if requested
                if (saveFrames && !string.IsNullOrEmpty(outputPath))
                {
                    bool saveSuccess = await _api.CreateVideoAsync(result, outputPath);
                    if (saveSuccess)
                    {
                        UpdateStatus($"Frames saved to {outputPath}_frame_*.png");
                    }
                }
                
                // Start playback
                if (result.GeneratedFrames != null && result.GeneratedFrames.Count > 0)
                {
                    StartPlayback(result.GeneratedFrames);
                }
            }
            else
            {
                UpdateStatus($"Generation failed: {result.ErrorMessage}");
            }
        }
        catch (Exception e)
        {
            UpdateStatus($"Error during generation: {e.Message}");
            Debug.LogError($"Generation error: {e}");
        }
        finally
        {
            if (generateButton) generateButton.interactable = true;
        }
    }
    
    async void OnClearCacheClicked()
    {
        if (!_initialized || _api == null)
            return;
            
        try
        {
            UpdateStatus("Clearing caches...");
            await _api.ClearCachesAsync();
            UpdateStatus($"Caches cleared\n{_api.GetCacheInfo()}");
        }
        catch (Exception e)
        {
            UpdateStatus($"Error clearing caches: {e.Message}");
            Debug.LogError($"Cache clearing error: {e}");
        }
    }
    
    void StartPlayback(System.Collections.Generic.List<Texture2D> frames)
    {
        if (_playbackCoroutine != null)
        {
            StopCoroutine(_playbackCoroutine);
        }
        
        _playbackCoroutine = StartCoroutine(PlayFrames(frames));
    }
    
    IEnumerator PlayFrames(System.Collections.Generic.List<Texture2D> frames)
    {
        if (outputImage == null || frames == null || frames.Count == 0)
            yield break;
            
        UpdateStatus($"Playing {frames.Count} frames...");
        
        float frameRate = 25f; // Target 25 FPS
        float frameDuration = 1f / frameRate;
        
        for (int i = 0; i < frames.Count; i++)
        {
            if (frames[i] != null)
            {
                outputImage.texture = frames[i];
            }
            
            yield return new WaitForSeconds(frameDuration);
        }
        
        UpdateStatus("Playback completed");
    }
    
    void UpdateStatus(string message)
    {
        if (statusText != null)
        {
            statusText.text = message;
        }
        
        Debug.Log($"[MuseTalkDemo] {message}");
    }
    
    void OnDestroy()
    {
        if (_playbackCoroutine != null)
        {
            StopCoroutine(_playbackCoroutine);
        }
        
        _api?.Dispose();
    }
} 