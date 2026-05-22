/*
 * ToolController.cs
 * ─────────────────────────────────────────────────────────
 * 역할: 손 데이터에 따라 3D 조리도구(식칼 / 휘핑기)를 이동시키고
 *       pinch gesture로 도구를 잡거나 놓는다.
 *
 * 씬 설정:
 *   - 각 조리도구 GameObject에 이 컴포넌트를 붙인다.
 *   - toolType으로 Knife / Whisk 중 선택.
 *   - 도구가 "잡힌 상태"일 때만 손 좌표를 따라간다.
 */

using UnityEngine;

public enum ToolType { Knife, Whisk }

public class ToolController : MonoBehaviour
{
    [Header("Tool Settings")]
    [SerializeField] private ToolType toolType = ToolType.Knife;

    [Tooltip("손과 도구 사이 최대 잡기 거리 (Unity 단위)")]
    [SerializeField] private float grabRadius = 1.5f;

    [Tooltip("도구 이동 스무딩 (낮을수록 빠름)")]
    [SerializeField] private float smoothing = 8f;

    // 도구가 잡혀있는지 여부 (외부에서 읽기 전용)
    public bool IsHeld  { get; private set; }
    public ToolType Type => toolType;

    private Vector3   _targetPos;
    private Renderer  _renderer;
    private Color     _baseColor;

    void Start()
    {
        _targetPos = transform.position;
        _renderer  = GetComponent<Renderer>();
        if (_renderer) _baseColor = _renderer.material.color;
    }

    void Update()
    {
        HandData hand = UDPReceiver.Current;

        if (!hand.detected)
        {
            IsHeld = false;
            _SetHighlight(false);
            return;
        }

        Vector3 handPos = new Vector3(hand.x, hand.y, 0f);
        float dist = Vector3.Distance(handPos, transform.position);

        // Pinch 시작 → 도구가 충분히 가까우면 잡기
        if (hand.pinched && dist < grabRadius)
            IsHeld = true;

        // Pinch 해제 → 놓기
        if (!hand.pinched)
            IsHeld = false;

        // 잡혀있으면 손 위치를 따라감
        if (IsHeld)
            _targetPos = handPos;

        // 스무딩 이동
        transform.position = Vector3.Lerp(
            transform.position, _targetPos, Time.deltaTime * smoothing);

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
