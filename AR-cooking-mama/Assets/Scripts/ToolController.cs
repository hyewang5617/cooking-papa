/*
 * ToolController.cs
 * ─────────────────────────────────────────────────────────
 * 역할: 손 데이터에 따라 3D 조리도구(식칼 / 밥주걱)를 이동시키고
 *       pinch gesture로 도구를 잡거나 놓는다.
 *
 * 씬 설정:
 *   - 각 조리도구 GameObject에 이 컴포넌트를 붙인다.
 *   - toolType으로 Knife / Paddle 중 선택.
 *   - 도구가 "잡힌 상태"일 때만 손 좌표를 따라간다.
 */

using UnityEngine;

public enum ToolType { Knife, Paddle }

public class ToolController : MonoBehaviour
{
    [Header("Tool Settings")]
    [SerializeField] private ToolType toolType = ToolType.Knife;

    public bool IsHeld   { get; private set; }
    public bool IsActive { get; set; } = true;   // set false to ignore hand input
    public ToolType Type => toolType;

    private Renderer _renderer;
    private Color    _baseColor;

    void Start()
    {
        _renderer = GetComponent<Renderer>();
        if (_renderer) _baseColor = _renderer.material.color;
    }

    void Update()
    {
        if (!IsActive) { IsHeld = false; _SetHighlight(false); return; }

        HandData hand = UDPReceiver.Current;

        if (hand.detected)
        {
            IsHeld = hand.pinched;
            if (IsHeld)
            {
                // 패킷 도착 위치 + 그 이후 경과 시간만큼 velocity로 예측
                float dt       = Time.time - UDPReceiver.LastUpdateTime;
                Vector3 base_  = new Vector3(hand.x, hand.y, hand.z);
                Vector3 pred   = new Vector3(hand.vx * dt, hand.vy * dt, 0f);
                transform.position = base_ + pred;
            }
        }
        else
        {
            IsHeld = false;
        }

        _SetHighlight(IsHeld);
    }

    private void _SetHighlight(bool on)
    {
        if (!_renderer) return;
        _renderer.material.color = on
            ? Color.Lerp(_baseColor, Color.yellow, 0.5f)
            : _baseColor;
    }
}
