/*
 * ScoreManager.cs
 * ─────────────────────────────────────────────────────────
 * 역할: 점수 계산 (정확도 + 속도 보너스 + 콤보 배수)
 *       싱글턴으로 어디서든 ScoreManager.Instance로 접근 가능.
 */

using UnityEngine;
using TMPro;  // 없으면 UnityEngine.UI.Text 로 교체

public class ScoreManager : MonoBehaviour
{
    public static ScoreManager Instance { get; private set; }

    [Header("UI")]
    [SerializeField] private TMP_Text scoreText;
    [SerializeField] private TMP_Text comboText;

    public int TotalScore { get; private set; }
    public int Combo      { get; private set; }

    void Awake()
    {
        if (Instance != null) { Destroy(gameObject); return; }
        Instance = this;
    }

    public void ResetScore()
    {
        TotalScore = 0;
        Combo      = 0;
        _UpdateUI();
    }

    public void AddScore(int points)
    {
        Combo++;
        float multiplier = 1f + (Combo / 4) * 0.25f;   // 4콤보마다 +25%
        multiplier = Mathf.Min(multiplier, 2.5f);
        int earned = Mathf.RoundToInt(points * multiplier);
        TotalScore += earned;
        _UpdateUI();
        Debug.Log($"[Score] +{earned} (x{multiplier:F2}) | Total={TotalScore} | Combo={Combo}");
    }

    public void ResetCombo()
    {
        Combo = 0;
        _UpdateUI();
    }

    // 점수 공식: 기본 100점 × 정확도 + 남은 시간 보너스
    public static int CalcScore(int achieved, int target, float timeLeft, float totalTime)
    {
        float accuracy   = Mathf.Clamp01((float)achieved / target);
        float speedBonus = Mathf.Clamp01(timeLeft / totalTime);
        return Mathf.RoundToInt(100f * accuracy + 50f * speedBonus);
    }

    private void _UpdateUI()
    {
        if (scoreText) scoreText.text = $"Score: {TotalScore}";
        if (comboText) comboText.text = Combo > 1 ? $"x{Combo} COMBO!" : "";
    }
}
