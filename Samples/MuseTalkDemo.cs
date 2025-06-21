using UnityEngine;
using UnityEngine.UI;
using System.Linq;
using MuseTalk;
using MuseTalk.Models;
using TMPro;

/// <summary>
/// Demo script showing how to use MuseTalk for talking head generation
/// </summary>
public class MuseTalkDemo : MonoBehaviour
{
    [Header("References")]
    public RawImage displayImage;
    public AudioSource audioSource;
    public Texture2D avatarTexture;
    public AudioClip testAudioClip;
    
    [Header("Avatar Folder (Optional - overrides single avatar)")]
    [Tooltip("Path to folder containing avatar images (relative to StreamingAssets)")]
    public string avatarFolderPath = "MuseTalk/avatars";
    [Tooltip("Supported image extensions")]
    public string[] supportedExtensions = { ".png", ".jpg", ".jpeg" };
    
    [Header("UI Elements")]
    public Button generateButton;
    public Button playVideoButton;
    public Button clearCacheButton;
    public Button reloadAvatarsButton;
    public TMP_Text statusText;
    public TMP_Text avatarInfoText;
    public Slider progressSlider;
    
    [Header("Settings")]
    public string modelPath = "MuseTalk";
    public string version = "v15";
    public int batchSize = 4;
    [Tooltip("Enable INT8 quantization for better performance")]
    public bool useInt8 = false;
    
    // MuseTalk components
    private MuseTalkFactory _factory;
    private CharacterTalker _characterTalker;
    private MuseTalkResult _lastResult;
    private bool _isGenerating = false;
    
    // Avatar management
    private Texture2D[] _avatarTextures;
    private bool _useMultipleAvatars = false;
    
    // Video playback
    private Coroutine _videoPlayback;
    private int _currentFrame = 0;
    
    void Start()
    {
        // Initialize UI
        SetupUI();
        
        // Initialize MuseTalk
        InitializeMuseTalk();
        
        // Load avatar textures
        LoadAvatarTextures();
        
        UpdateStatus("Ready to generate talking head video");
    }
    
    void SetupUI()
    {
        if (generateButton != null)
        {
            generateButton.onClick.AddListener(GenerateTalkingHead);
        }
        
        if (playVideoButton != null)
        {
            playVideoButton.onClick.AddListener(PlayGeneratedVideo);
            playVideoButton.interactable = false;
        }
        
        if (clearCacheButton != null)
        {
            clearCacheButton.onClick.AddListener(ClearCache);
        }
        
        if (reloadAvatarsButton != null)
        {
            reloadAvatarsButton.onClick.AddListener(ReloadAvatars);
        }
        
        if (progressSlider != null)
        {
            progressSlider.value = 0f;
        }
    }
    
    void LoadAvatarTextures()
    {
        try
        {
            // First try to load from folder if specified
            if (!string.IsNullOrEmpty(avatarFolderPath))
            {
                string fullFolderPath = System.IO.Path.Combine(Application.streamingAssetsPath, avatarFolderPath);
                
                if (System.IO.Directory.Exists(fullFolderPath))
                {
                    var avatarList = new System.Collections.Generic.List<Texture2D>();
                    
                    // Get all image files from folder
                    foreach (string extension in supportedExtensions)
                    {
                        string[] files = System.IO.Directory.GetFiles(fullFolderPath, "*" + extension, System.IO.SearchOption.TopDirectoryOnly);
                        
                        foreach (string filePath in files)
                        {
                            try
                            {
                                byte[] fileData = System.IO.File.ReadAllBytes(filePath);
                                Texture2D texture = new Texture2D(2, 2);
                                
                                if (texture.LoadImage(fileData))
                                {
                                    texture.name = System.IO.Path.GetFileNameWithoutExtension(filePath);
                                    
                                    // Ensure texture is in RGB24 format for consistency with single avatar texture
                                    if (texture.format != TextureFormat.RGB24)
                                    {
                                        var convertedTexture = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false);
                                        convertedTexture.name = texture.name;
                                        convertedTexture.SetPixels(texture.GetPixels());
                                        convertedTexture.Apply();
                                        
                                        // Destroy the original texture and use the converted one
                                        DestroyImmediate(texture);
                                        texture = convertedTexture;
                                    }
                                    
                                    avatarList.Add(texture);
                                    Debug.Log($"Loaded avatar image: {texture.name} ({texture.format}, {texture.width}x{texture.height})");
                                }
                                else
                                {
                                    Debug.LogWarning($"Failed to load image: {filePath}");
                                    DestroyImmediate(texture);
                                }
                            }
                            catch (System.Exception e)
                            {
                                Debug.LogError($"Error loading avatar image {filePath}: {e.Message}");
                            }
                        }
                    }
                    
                    if (avatarList.Count > 0)
                    {
                        // Sort avatars by name (handles numbered sequences like 00000000, 00000001, etc.)
                        avatarList.Sort((a, b) => string.Compare(a.name, b.name, System.StringComparison.Ordinal));
                        
                        _avatarTextures = avatarList.ToArray();
                        _useMultipleAvatars = true;
                        UpdateStatus($"Loaded {_avatarTextures.Length} avatar images from folder (sorted by name)");
                        UpdateAvatarInfo();
                        return;
                    }
                    else
                    {
                        UpdateStatus($"No valid images found in folder: {fullFolderPath}");
                    }
                }
                else
                {
                    UpdateStatus($"Avatar folder not found: {fullFolderPath}");
                }
            }
            
            // Fallback to single avatar texture
            if (avatarTexture != null)
            {
                _avatarTextures = new Texture2D[] { avatarTexture };
                _useMultipleAvatars = false;
                UpdateStatus($"Using single avatar texture: {avatarTexture.name}");
                UpdateAvatarInfo();
            }
            else
            {
                UpdateStatus("No avatar textures available");
                UpdateAvatarInfo();
            }
        }
        catch (System.Exception e)
        {
            UpdateStatus($"Error loading avatar textures: {e.Message}");
            Debug.LogError($"Avatar loading error: {e}");
            
            // Fallback to single avatar
            if (avatarTexture != null)
            {
                _avatarTextures = new Texture2D[] { avatarTexture };
                _useMultipleAvatars = false;
            }
        }
    }
    
    void InitializeMuseTalk()
    {
        try
        {
            modelPath = Application.streamingAssetsPath + "/" + modelPath;
            // Create configuration
            var config = new MuseTalkConfig(modelPath, version)
            {
                BatchSize = batchSize,
                Device = "cpu", // Use CPU for compatibility
                UseINT8 = useInt8
            };
            
            // Initialize factory
            _factory = MuseTalkFactory.Instance;
            _factory.EnableLogging = true;
            _factory.EnableFileDebug = true;
            
            bool initialized = _factory.Initialize(config);
            
            if (initialized)
            {
                UpdateStatus("MuseTalk initialized successfully");
                
                // Create character talker if avatar is provided
                if (avatarTexture != null)
                {
                    CreateCharacterTalker();
                }
            }
            else
            {
                UpdateStatus("Failed to initialize MuseTalk - check model path and files");
                Debug.LogError("MuseTalk initialization failed. Make sure ONNX models are in the correct path.");
            }
        }
        catch (System.Exception e)
        {
            UpdateStatus($"Error initializing MuseTalk: {e.Message}");
            Debug.LogError($"MuseTalk initialization error: {e}");
        }
    }
    
    void CreateCharacterTalker()
    {
        if (avatarTexture == null || _factory == null || !_factory.IsReady)
            return;
            
        try
        {
            // Dispose previous talker if any
            _characterTalker?.Dispose();
            
            // Create new character talker
            var talkerConfig = new MuseTalkConfig(modelPath, version)
            {
                BatchSize = batchSize,
                UseINT8 = useInt8
            };
            
            // This would need to be added to MuseTalkFactory
            // _characterTalker = _factory.CreateCharacterTalker(avatarTexture, talkerConfig);
            
            UpdateStatus($"Character talker created for {avatarTexture.name}");
        }
        catch (System.Exception e)
        {
            UpdateStatus($"Error creating character talker: {e.Message}");
            Debug.LogError($"Character talker creation error: {e}");
        }
    }
    
    async void GenerateTalkingHead()
    {
        if (_isGenerating)
        {
            UpdateStatus("Already generating...");
            return;
        }
        
        if (!_factory.IsReady)
        {
            UpdateStatus("MuseTalk factory not ready");
            return;
        }
        
        if (_avatarTextures == null || _avatarTextures.Length == 0)
        {
            UpdateStatus("No avatar textures available");
            return;
        }
        
        if (testAudioClip == null)
        {
            UpdateStatus("No audio clip provided");
            return;
        }
        
        _isGenerating = true;
        SetButtonsInteractable(false);
        UpdateStatus("Generating talking head video...");
        
        try
        {
            // Reset progress
            if (progressSlider != null)
                progressSlider.value = 0f;
            
            // Generate talking head video using factory with multiple avatars
            UpdateStatus($"Generating with {_avatarTextures.Length} avatar texture(s)...");
            _lastResult = await _factory.GenerateAsync(_avatarTextures, testAudioClip, batchSize);
            
            if (_lastResult.Success)
            {
                UpdateStatus($"Generated {_lastResult.FrameCount} frames");
                
                if (playVideoButton != null)
                    playVideoButton.interactable = true;
                    
                // Show first frame
                if (_lastResult.GeneratedFrames.Count > 0 && displayImage != null)
                {
                    displayImage.texture = _lastResult.GeneratedFrames[0];
                }
                
                if (progressSlider != null)
                    progressSlider.value = 1f;
            }
            else
            {
                UpdateStatus($"Generation failed: {_lastResult.ErrorMessage}");
            }
        }
        catch (System.Exception e)
        {
            UpdateStatus($"Error during generation: {e.Message}");
            Debug.LogError($"MuseTalk generation error: {e}");
        }
        finally
        {
            _isGenerating = false;
            SetButtonsInteractable(true);
        }
    }
    
    void PlayGeneratedVideo()
    {
        if (_lastResult == null || !_lastResult.Success || _lastResult.GeneratedFrames.Count == 0)
        {
            UpdateStatus("No video to play");
            return;
        }
        
        // Stop any existing playback
        if (_videoPlayback != null)
        {
            StopCoroutine(_videoPlayback);
        }
        
        // Start video playback
        _videoPlayback = StartCoroutine(PlayVideoCoroutine());
        
        // Play audio
        if (audioSource != null && testAudioClip != null)
        {
            audioSource.clip = testAudioClip;
            audioSource.Play();
        }
    }
    
    System.Collections.IEnumerator PlayVideoCoroutine()
    {
        if (_lastResult?.GeneratedFrames == null)
            yield break;
            
        UpdateStatus("Playing generated video...");
        
        float fps = 25f; // Standard video FPS
        float frameTime = 1f / fps;
        
        _currentFrame = 0;
        
        while (_currentFrame < _lastResult.GeneratedFrames.Count)
        {
            // Display current frame
            if (displayImage != null && _lastResult.GeneratedFrames[_currentFrame] != null)
            {
                displayImage.texture = _lastResult.GeneratedFrames[_currentFrame];
            }
            
            // Update progress
            if (progressSlider != null)
            {
                progressSlider.value = (float)_currentFrame / _lastResult.GeneratedFrames.Count;
            }
            
            _currentFrame++;
            yield return new WaitForSeconds(frameTime);
        }
        
        UpdateStatus($"Video playback completed ({_lastResult.GeneratedFrames.Count} frames)");
        _videoPlayback = null;
    }
    
    void ClearCache()
    {
        _characterTalker?.ClearCache();
        
        if (_lastResult != null)
        {
            // Clean up generated frames
            foreach (var frame in _lastResult.GeneratedFrames)
            {
                if (frame != null)
                    DestroyImmediate(frame);
            }
            _lastResult = null;
        }
        
        if (playVideoButton != null)
            playVideoButton.interactable = false;
            
        if (displayImage != null)
            displayImage.texture = null;
            
        UpdateStatus("Cache cleared");
    }
    
    void ReloadAvatars()
    {
        // Clean up existing textures if they were loaded from folder
        if (_useMultipleAvatars && _avatarTextures != null)
        {
            foreach (var tex in _avatarTextures)
            {
                if (tex != null && tex != avatarTexture) // Don't destroy the inspector-assigned texture
                {
                    DestroyImmediate(tex);
                }
            }
        }
        
        // Reload avatar textures
        LoadAvatarTextures();
        UpdateStatus("Avatar textures reloaded");
    }
    
    void UpdateAvatarInfo()
    {
        if (avatarInfoText == null) return;
        
        if (_avatarTextures == null || _avatarTextures.Length == 0)
        {
            avatarInfoText.text = "No avatars loaded";
        }
        else if (_useMultipleAvatars)
        {
            string[] names = _avatarTextures.Select(t => t?.name ?? "Unknown").ToArray();
            avatarInfoText.text = $"Multiple Avatars ({_avatarTextures.Length}):\n{string.Join(", ", names)}";
        }
        else
        {
            avatarInfoText.text = $"Single Avatar: {_avatarTextures[0]?.name ?? "Unknown"}";
        }
    }
    
    void SetButtonsInteractable(bool interactable)
    {
        if (generateButton != null) generateButton.interactable = interactable;
        if (clearCacheButton != null) clearCacheButton.interactable = interactable;
        if (reloadAvatarsButton != null) reloadAvatarsButton.interactable = interactable;
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
        // Stop video playback
        if (_videoPlayback != null)
        {
            StopCoroutine(_videoPlayback);
        }
        
        // Clean up resources
        _characterTalker?.Dispose();
        ClearCache();
        
        // Don't dispose factory as it's a singleton
    }
    
    // Validation in editor
    void OnValidate()
    {
        if (avatarTexture != null)
        {
            // Check if avatar texture is readable
            try
            {
                var pixels = avatarTexture.GetPixels(0, 0, 1, 1);
            }
            catch
            {
                Debug.LogWarning("Avatar texture is not readable. Please enable 'Read/Write' in import settings.");
            }
        }
        
        // Validate avatar folder path
        if (!string.IsNullOrEmpty(avatarFolderPath))
        {
            string fullPath = System.IO.Path.Combine(Application.streamingAssetsPath, avatarFolderPath);
            if (Application.isPlaying && !System.IO.Directory.Exists(fullPath))
            {
                Debug.LogWarning($"Avatar folder does not exist: {fullPath}. Create the folder and add avatar images, or leave avatarFolderPath empty to use single avatar texture.");
            }
        }
    }
} 