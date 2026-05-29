using UnityEngine;

public enum ToolType { Knife, Paddle }

public class ToolController : MonoBehaviour
{
    [SerializeField] private ToolType toolType;

    public ToolType Type   => toolType;
    public bool     IsHeld { get; private set; }

    void Update()
    {
        HandData h = UDPReceiver.Current;
        IsHeld = h.detected && h.pinched;
        if (IsHeld)
            transform.position = new Vector3(h.x, h.y, h.z);
    }

    void OnDisable() => IsHeld = false;
}
