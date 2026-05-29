using UnityEngine;
using UnityEngine.Events;

public class WhiskingGameManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private ToolController paddle;
    [SerializeField] private GameObject     bowl;

    [Header("Game Settings")]
    [SerializeField] private int   targetRotations = 5;
    [SerializeField] private float gameDuration    = 25f;

    [Header("Events")]
    public UnityEvent onSuccess;
    public UnityEvent onFail;

    public int   RotationCount { get; private set; }
    public float GaugeProgress { get; private set; }
    public float TimeLeft      { get; private set; }
    public bool  IsRunning     { get; private set; }

    private float   _totalAngle = 0f;
    private float   _prevAngle  = float.NaN;
    private Vector2 _center;

    public void StartGame()
    {
        RotationCount = 0;
        GaugeProgress = 0f;
        TimeLeft      = gameDuration;
        IsRunning     = true;
        _totalAngle   = 0f;
        _prevAngle    = float.NaN;
        if (paddle) paddle.enabled = true;

        _center = bowl
            ? new Vector2(bowl.transform.position.x, bowl.transform.position.y)
            : Vector2.zero;

        Debug.Log("[PaddlingGame] Started");
    }

    void Update()
    {
        if (!IsRunning) return;

        TimeLeft -= Time.deltaTime;
        if (TimeLeft <= 0f) { _Fail(); return; }

        HandData hand = UDPReceiver.Current;
        if (!hand.detected || !paddle || !paddle.IsHeld)
        {
            _prevAngle = float.NaN;
            return;
        }

        Vector2 dir = new Vector2(hand.x, hand.y) - _center;
        if (dir.magnitude < 0.3f) return;

        float angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;

        if (!float.IsNaN(_prevAngle))
        {
            _totalAngle += Mathf.DeltaAngle(_prevAngle, angle);

            int newRot = Mathf.FloorToInt(Mathf.Abs(_totalAngle) / 360f);
            if (newRot > RotationCount)
            {
                RotationCount = newRot;
                Debug.Log($"[PaddlingGame] Rotation! ({RotationCount}/{targetRotations})");
                if (RotationCount >= targetRotations) { _Success(); return; }
            }

            GaugeProgress = (Mathf.Abs(_totalAngle) % 360f) / 360f;
        }

        _prevAngle = angle;
    }

    private void _Success()
    {
        IsRunning = false;
        if (paddle) paddle.enabled = false;
        ScoreManager.Instance?.AddScore(
            ScoreManager.CalcScore(RotationCount, targetRotations, TimeLeft, gameDuration));
        onSuccess?.Invoke();
        Debug.Log("[PaddlingGame] SUCCESS");
    }

    private void _Fail()
    {
        IsRunning = false;
        if (paddle) paddle.enabled = false;
        onFail?.Invoke();
        Debug.Log("[PaddlingGame] FAIL");
    }
}
