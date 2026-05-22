/*
 * WhiskingGameManager.cs
 * ─────────────────────────────────────────────────────────
 * 역할: 휘핑 미션을 관리한다.
 *
 * 판정 로직:
 *   - 휘핑기(ToolType.Whisk)가 잡혀있는 동안 손 궤적의 각도 변화를 추적
 *   - 360도 누적 회전마다 게이지 +1
 *   - 게이지가 목표 이상이면 성공
 *
 * 씬 설정:
 *   - 빈 GameObject에 부착
 *   - whisk 필드에 휘핑기 오브젝트(ToolController 부착) 연결
 *   - bowl 필드에 그릇 오브젝트 연결 (선택)
 */

using UnityEngine;
using UnityEngine.Events;

public class WhiskingGameManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private ToolController whisk;
    [SerializeField] private GameObject     bowl;

    [Header("Game Settings")]
    [SerializeField] private int   targetRotations = 5;
    [SerializeField] private float gameDuration    = 25f;

    [Header("Events")]
    public UnityEvent onSuccess;
    public UnityEvent onFail;

    public int   RotationCount  { get; private set; }
    public float GaugeProgress  { get; private set; }  // 0~1, 현재 회전 진행률
    public float TimeLeft       { get; private set; }
    public bool  IsRunning      { get; private set; }

    private float  _totalAngle  = 0f;
    private float  _prevAngle   = float.NaN;
    private Vector2 _center;    // 그릇(또는 화면) 중심

    public void StartGame()
    {
        RotationCount = 0;
        GaugeProgress = 0f;
        TimeLeft      = gameDuration;
        IsRunning     = true;
        _totalAngle   = 0f;
        _prevAngle    = float.NaN;

        // 중심: 그릇 위치 또는 월드 원점
        _center = bowl
            ? new Vector2(bowl.transform.position.x, bowl.transform.position.y)
            : Vector2.zero;

        Debug.Log("[WhiskingGame] Started");
    }

    void Update()
    {
        if (!IsRunning) return;

        TimeLeft -= Time.deltaTime;
        if (TimeLeft <= 0f) { _Fail(); return; }

        HandData hand = UDPReceiver.Current;
        if (!hand.detected || !whisk || !whisk.IsHeld)
        {
            _prevAngle = float.NaN;
            return;
        }

        Vector2 handPos = new Vector2(hand.x, hand.y);
        Vector2 dir     = handPos - _center;

        // 중심에 너무 가까우면 무시
        if (dir.magnitude < 0.3f) return;

        float angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;

        if (!float.IsNaN(_prevAngle))
        {
            float delta = Mathf.DeltaAngle(_prevAngle, angle);
            _totalAngle += delta;

            int newRot = Mathf.FloorToInt(Mathf.Abs(_totalAngle) / 360f);
            if (newRot > RotationCount)
            {
                RotationCount = newRot;
                Debug.Log($"[WhiskingGame] Rotation! ({RotationCount}/{targetRotations})");
                if (RotationCount >= targetRotations) { _Success(); return; }
            }

            GaugeProgress = (Mathf.Abs(_totalAngle) % 360f) / 360f;
        }

        _prevAngle = angle;
    }

    private void _Success()
    {
        IsRunning = false;
        ScoreManager.Instance?.AddScore(
            ScoreManager.CalcScore(RotationCount, targetRotations, TimeLeft, gameDuration));
        onSuccess?.Invoke();
        Debug.Log("[WhiskingGame] SUCCESS");
    }

    private void _Fail()
    {
        IsRunning = false;
        onFail?.Invoke();
        Debug.Log("[WhiskingGame] FAIL");
    }
}
