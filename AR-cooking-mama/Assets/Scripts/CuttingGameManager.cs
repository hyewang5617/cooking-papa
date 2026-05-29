using UnityEngine;
using UnityEngine.Events;

public class CuttingGameManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private ToolController knife;
    [SerializeField] private GameObject     ingredient;

    [Header("Game Settings")]
    [SerializeField] private int   targetCuts        = 8;
    [SerializeField] private float gameDuration      = 20f;
    [SerializeField] private float cutSpeedThreshold = 3f;

    [Header("Events")]
    public UnityEvent onSuccess;
    public UnityEvent onFail;

    public int   CutCount  { get; private set; }
    public float TimeLeft  { get; private set; }
    public bool  IsRunning { get; private set; }

    private float _prevVy   = 0f;
    private int   _cooldown = 0;

    public void StartGame()
    {
        CutCount  = 0;
        TimeLeft  = gameDuration;
        IsRunning = true;
        _prevVy   = 0f;
        if (knife)      knife.enabled = true;
        if (ingredient) ingredient.SetActive(true);
        Debug.Log("[CuttingGame] Started");
    }

    void Update()
    {
        if (!IsRunning) return;

        TimeLeft -= Time.deltaTime;
        if (TimeLeft <= 0f) { _Fail(); return; }

        if (_cooldown > 0) { _cooldown--; return; }

        HandData hand = UDPReceiver.Current;

        if (!knife || !knife.IsHeld) { _prevVy = hand.vy; return; }

        float vy = hand.vy;

        bool dirChanged = (_prevVy > cutSpeedThreshold && vy < -cutSpeedThreshold)
                       || (_prevVy < -cutSpeedThreshold && vy > cutSpeedThreshold);

        if (dirChanged)
        {
            CutCount++;
            _cooldown = 6;
            _AnimateCut();
            Debug.Log($"[CuttingGame] Cut! ({CutCount}/{targetCuts})");
            if (CutCount >= targetCuts) _Success();
        }

        _prevVy = vy;
    }

    private void _AnimateCut()
    {
        if (ingredient) ingredient.transform.localScale = Vector3.one * 0.85f;
        Invoke(nameof(_ResetScale), 0.08f);
    }

    private void _ResetScale()
    {
        if (ingredient) ingredient.transform.localScale = Vector3.one;
    }

    private void _Success()
    {
        IsRunning = false;
        if (knife) knife.enabled = false;
        ScoreManager.Instance?.AddScore(ScoreManager.CalcScore(CutCount, targetCuts, TimeLeft, gameDuration));
        onSuccess?.Invoke();
        Debug.Log("[CuttingGame] SUCCESS");
    }

    private void _Fail()
    {
        IsRunning = false;
        if (knife) knife.enabled = false;
        onFail?.Invoke();
        Debug.Log("[CuttingGame] FAIL");
    }
}
