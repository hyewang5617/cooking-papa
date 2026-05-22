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

    [Tooltip("도구가 목표 위치까지 도달하는 시간(초). 낮을수록 빠름")]
    [SerializeField] private float smoothTime = 0.08f;

    [Tooltip("도구 최대 이동 속도 (Unity 단위/초)")]
    [SerializeField] private float maxSpeed = 30f;

    public bool IsHeld  { get; private set; }
    public ToolType Type => toolType;

    private Vector3  _targetPos;
    private Vector3  _velocity = Vector3.zero;   // SmoothDamp 내부 속도
    private Renderer _renderer;
    private Color    _baseColor;

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

        // pinch 여부만으로 잡기/놓기 결정 (거리 제한 없음)
        IsHeld = hand.pinched;

        if (IsHeld)
            _targetPos = handPos;

        transform.position = Vector3.SmoothDamp(
            transform.position, _targetPos, ref _velocity, smoothTime, maxSpeed);

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
