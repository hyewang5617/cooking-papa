/*
 * WhiskingGameManager.cs  (밥주걱 버전 - Paddle)
 * ─────────────────────────────────────────────────────────
 * 역할: 밥주걱(Paddle) 미션을 관리한다.
 *
 * 판정 로직:
 *   - 밥주걱(ToolType.Paddle)이 잡혀있는 동안 손 궤적의 각도 변화를 추적
 *   - 360도 누적 회전마다 게이지 +1
 *   - 게이지가 목표 이상이면 성공
 *
 * 씬 설정:
 *   - 빈 GameObject에 부착
 *   - paddle 필드에 밥주걱 오브젝트(ToolController 부착) 연결
 *   - bowl 필드에 그릇 오브젝트 연결 (선택)
 */

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

    public int   RotationCount  { get; private set; }
    public float GaugeProgress  { get; private set; }  // 0~1, 현재 회전 진행률
    public float TimeLeft       { get; private set; }
    public bool  IsRunning      { get; private set; }

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
        if (paddle) paddle.IsActive = true;

        // 중심: 그릇 위치 또는 월드 원점
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

        Vector2 handPos = new Vector2(hand.x, hand.y);
        Vector2 dir     = handPos - _center;

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
        if (paddle) paddle.IsActive = false;
        ScoreManager.Instance?.AddScore(
            ScoreManager.CalcScore(RotationCount, targetRotations, TimeLeft, gameDuration));
        onSuccess?.Invoke();
        Debug.Log("[PaddlingGame] SUCCESS");
    }

    private void _Fail()
    {
        IsRunning = false;
        if (paddle) paddle.IsActive = false;
        onFail?.Invoke();
        Debug.Log("[PaddlingGame] FAIL");
    }
}
