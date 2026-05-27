/*
 * UDPReceiver.cs
 * ─────────────────────────────────────────────────────────
 * 역할: Python에서 UDP로 전송된 손 데이터를 수신하고
 *       HandData 구조체로 파싱한 뒤 다른 스크립트에 공유한다.
 *
 * 실행 순서:
 *   1. Start() → 백그라운드 스레드에서 UDP 수신 시작
 *   2. 데이터 수신 시 → JSON 파싱 → _latest에 저장
 *   3. Update() → 메인 스레드에서 _latest를 Current로 복사 (스레드 안전)
 */

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class HandData
{
    public bool   detected;
    public float  x;           // Unity 월드 X 좌표
    public float  y;           // Unity 월드 Y 좌표
    public float  z;           // Unity 월드 Z 좌표 (손 깊이)
    public float  vx;          // X 속도 (Unity 단위/초)
    public float  vy;          // Y 속도
    public bool   pinched;     // pinch 여부
    public float  pinch_dist;  // 엄지-검지 거리 (정규화)
}

public class UDPReceiver : MonoBehaviour
{
    [Header("Network")]
    [SerializeField] private int port = 5052;

    public static HandData Current { get; private set; } = new HandData();

    private UdpClient  _client;
    private Thread     _thread;
    private HandData   _latest = new HandData();
    private bool       _hasNew = false;
    private readonly object _lock = new object();

    void Start()
    {
        Current = new HandData();
        _client = new UdpClient(port);
        _thread = new Thread(ReceiveLoop) { IsBackground = true };
        _thread.Start();
        Debug.Log($"[UDPReceiver] Listening on port {port}");
    }

    void Update()
    {
        // 백그라운드 스레드에서 받은 데이터를 메인 스레드로 안전하게 복사
        lock (_lock)
        {
            if (_hasNew)
            {
                Current = _latest;
                _hasNew = false;
            }
        }
    }

    private void ReceiveLoop()
    {
        IPEndPoint ep = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try
            {
                byte[] bytes = _client.Receive(ref ep);
                string json  = Encoding.UTF8.GetString(bytes);
                HandData data = JsonUtility.FromJson<HandData>(json);
                lock (_lock)
                {
                    _latest = data;
                    _hasNew = true;
                }
            }
            catch (SocketException) { break; }
            catch (Exception e) { Debug.LogWarning($"[UDPReceiver] {e.Message}"); }
        }
    }

    void OnDestroy()
    {
        _client?.Close();
        _thread?.Abort();
    }
}
