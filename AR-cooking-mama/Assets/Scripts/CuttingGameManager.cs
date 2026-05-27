/*
 * CuttingGameManager.cs
 * ─────────────────────────────────────────────────────────
 * 역할: 칼질 미션을 관리한다.
 *
 * 판정 로직:
 *   - 식칼(ToolType.Knife)이 잡혀있는 동안 Y 속도를 모니터링
 *   - |vy| > cutSpeedThreshold 이면 '1회 칼질' 판정
 *   - 목표 횟수 달성 → 성공, 시간 초과 → 실패
 *
 * 씬 설정:
 *   - 빈 GameObject에 부착
 *   - knife 필드에 식칼 오브젝트(ToolController 부착) 연결
 *   - ingredient 필드에 재료 오브젝트 연결
 *   - UI Text는 ScoreManager가 관리
 */

using UnityEngine;
using UnityEngine.Events;

public class CuttingGameManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private ToolController knife;
    [SerializeField] private GameObject     ingredient;   // 썰릴 재료

    [Header("Game Settings")]
    [SerializeField] private int   targetCuts     = 8;
    [SerializeField] private float gameDuration   = 20f;
    [SerializeField] private float cutSpeedThreshold = 3f;  // Unity 단위/초

    [Header("Events")]
    public UnityEvent onSuccess;
    public UnityEvent onFail;

    public int   CutCount    { get; private set; }
    public float TimeLeft    { get; private set; }
    public bool  IsRunning   { get; private set; }

    private float _prevVy     = 0f;
    private int   _cooldown   = 0;   // 연속 중복 판정 방지 (프레임 단위)

    public void StartGame()
    {
        CutCount  = 0;
        TimeLeft  = gameDuration;
        IsRunning = true;
        _prevVy   = 0f;
        if (knife)      knife.IsActive = true;
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

        // 칼이 잡혀있을 때만 판정
        if (!knife || !knife.IsHeld) { _prevVy = hand.vy; return; }

        float vy = hand.vy;

        // Y 방향 전환 + 속도가 임계값 초과 → 칼질 1회
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
        // 재료를 Y축으로 살짝 흔들어 시각 피드백
        if (ingredient)
            ingredient.transform.localScale = Vector3.one * 0.85f;
        Invoke(nameof(_ResetScale), 0.08f);
    }

    private void _ResetScale()
    {
        if (ingredient) ingredient.transform.localScale = Vector3.one;
    }

    private void _Success()
    {
        IsRunning = false;
        if (knife) knife.IsActive = false;
        ScoreManager.Instance?.AddScore(ScoreManager.CalcScore(CutCount, targetCuts, TimeLeft, gameDuration));
        onSuccess?.Invoke();
        Debug.Log("[CuttingGame] SUCCESS");
    }

    private void _Fail()
    {
        IsRunning = false;
        if (knife) knife.IsActive = false;
        onFail?.Invoke();
        Debug.Log("[CuttingGame] FAIL");
    }
}
