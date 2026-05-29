using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class HandData
{
    public bool  detected;
    public float x, y, z;
    public float vx, vy;
    public bool  pinched;
    public float pinch_dist;
}

public class UDPReceiver : MonoBehaviour
{
    [SerializeField] private int port = 5052;

    public static HandData Current { get; private set; } = new HandData();

    private UdpClient     _client;
    private Thread        _thread;
    private HandData      _pending;
    private volatile bool _hasNew;

    void Start()
    {
        Current = new HandData();
        _client = new UdpClient(port);
        _thread = new Thread(ReceiveLoop) { IsBackground = true };
        _thread.Start();
        Debug.Log($"[UDPReceiver] port {port}");
    }

    // 메인 스레드: 새 데이터 있으면 교체 (lock 없이 volatile로 동기화)
    void Update()
    {
        if (_hasNew) { Current = _pending; _hasNew = false; }
    }

    // 백그라운드 스레드: UDP 수신 → _pending에 파싱
    // .NET에서 참조 할당은 원자적이므로 volatile bool만으로 충분
    void ReceiveLoop()
    {
        var ep = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try
            {
                var bytes = _client.Receive(ref ep);
                _pending = JsonUtility.FromJson<HandData>(Encoding.UTF8.GetString(bytes));
                _hasNew  = true;
            }
            catch (SocketException) { break; }
            catch (Exception e)     { Debug.LogWarning($"[UDPReceiver] {e.Message}"); }
        }
    }

    void OnDestroy() => _client?.Close();
}
