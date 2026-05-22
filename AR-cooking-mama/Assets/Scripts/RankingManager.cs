/*
 * RankingManager.cs
 * ─────────────────────────────────────────────────────────
 * 역할: 게임 종료 시 점수를 JSON 파일에 저장하고 Top 10 랭킹을 표시.
 *
 * 저장 경로: Application.persistentDataPath/rankings.json
 *   Windows: C:\Users\<user>\AppData\LocalLow\<company>\<product>\rankings.json
 */

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using TMPro;

[Serializable]
public class RankEntry
{
    public string name;
    public int    score;
    public string date;
}

[Serializable]
class RankingData { public List<RankEntry> entries = new(); }

public class RankingManager : MonoBehaviour
{
    public static RankingManager Instance { get; private set; }

    [Header("UI")]
    [SerializeField] private TMP_Text rankingText;     // 랭킹 목록 표시
    [SerializeField] private TMP_InputField nameInput; // 이름 입력

    private RankingData _data = new();
    private string      _filePath;

    void Awake()
    {
        if (Instance != null) { Destroy(gameObject); return; }
        Instance  = this;
        _filePath = Path.Combine(Application.persistentDataPath, "rankings.json");
        _Load();
    }

    // 점수 저장 (이름 입력 후 호출)
    public void SaveScore()
    {
        string playerName = nameInput ? nameInput.text.Trim() : "Player";
        if (string.IsNullOrEmpty(playerName)) playerName = "Player";

        _data.entries.Add(new RankEntry
        {
            name  = playerName,
            score = ScoreManager.Instance ? ScoreManager.Instance.TotalScore : 0,
            date  = DateTime.Now.ToString("yyyy-MM-dd HH:mm"),
        });

        // 내림차순 정렬, Top 10만 유지
        _data.entries.Sort((a, b) => b.score.CompareTo(a.score));
        if (_data.entries.Count > 10)
            _data.entries.RemoveRange(10, _data.entries.Count - 10);

        _Save();
        ShowRanking();
    }

    public void ShowRanking()
    {
        if (!rankingText) return;
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("=== RANKING ===");
        for (int i = 0; i < _data.entries.Count; i++)
        {
            var e = _data.entries[i];
            sb.AppendLine($"{i + 1,2}. {e.name,-12} {e.score,6}  {e.date}");
        }
        rankingText.text = sb.ToString();
    }

    private void _Load()
    {
        if (File.Exists(_filePath))
        {
            try { _data = JsonUtility.FromJson<RankingData>(File.ReadAllText(_filePath)); }
            catch { _data = new RankingData(); }
        }
    }

    private void _Save()
    {
        File.WriteAllText(_filePath, JsonUtility.ToJson(_data, prettyPrint: true));
        Debug.Log($"[Ranking] Saved to {_filePath}");
    }
}
