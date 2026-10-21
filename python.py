import time
import struct
import socket
import fcntl

import numpy as np
from gnuradio import gr


class zynq_raw_eth_tx(gr.sync_block):
    """
    ZC702 Raw Ethernet TX block - FLOAT version for LMS_no_UDP_NEON.c

    The PC sends the raw complex64 samples. The ARM (NEON) on the board converts
    them to Q1.15, exactly like the C code does. No Q1.15 conversion here.

    Input 0: x(n) complex64
    Input 1: d(n) complex64

    Frame on the wire:
        14 B  Ethernet : dst MAC | src MAC | EtherType
        16 B  app hdr  : u32 seq | u32 N | u32 mu_q15 | u32 reserved   (little endian)
        8*N B          : x[N] complex64  (re0, im0, re1, im1, ...)
        8*N B          : d[N] complex64

    Frame size = 30 + 16*N bytes; the Ethernet payload is 16 + 16*N bytes.
        N = 64   -> 1054 B frame (fits MTU 1500, no jumbo frames needed)
        N = 512  -> 8222 B frame (needs MTU >= 8208 on this NIC: sudo ip link set <iface> mtu 9000)
    The board accepts N up to 574.

    Keep |re| and |im| of x and d below 1.0: the board saturates at +-1.0.
    """

    def __init__(
        self,
        iface='enp7s0',
        dest_mac='00:0a:35:00:01:03',
        ethertype=0x88B5,
        samples_per_frame=64,
        mu_q15=163,
        frame_gap_us=0.0,
    ):
        gr.sync_block.__init__(
            self,
            name="ZC702 Raw Ethernet LMS TX (float)",
            in_sig=[np.complex64, np.complex64],
            out_sig=None,
        )

        self.iface = str(iface)
        self.dest_mac_str = str(dest_mac)
        self.ethertype = int(ethertype)
        self.n = int(samples_per_frame)
        self.mu_q15 = int(mu_q15) & 0xFFFF
        self.gap = float(frame_gap_us) * 1e-6

        if not 1 <= self.n <= 574:
            raise ValueError("samples_per_frame must be 1..574")

        try:
            self.dest_mac = bytes.fromhex(
                self.dest_mac_str.replace(':', '').replace('-', ''))
        except Exception as ex:
            raise ValueError(f"Invalid destination MAC: {dest_mac}") from ex
        if len(self.dest_mac) != 6:
            raise ValueError("Destination MAC must contain exactly 6 bytes")

        self.src_mac = self.get_interface_mac(self.iface)
        self.eth_header = (self.dest_mac + self.src_mac
                           + struct.pack(">H", self.ethertype))

        # The frame must fit the interface MTU, otherwise send() fails.
        eth_payload = 16 + 16 * self.n
        mtu = self.get_interface_mtu(self.iface)
        if eth_payload > mtu:
            raise RuntimeError(
                f"N={self.n} needs MTU >= {eth_payload} but {self.iface} has "
                f"MTU {mtu}. Use samples_per_frame <= {(mtu - 16) // 16} or run: "
                f"sudo ip link set {self.iface} mtu 9000")

        # Always hand work() whole frames.
        self.set_output_multiple(self.n)

        self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW,
                                  socket.htons(self.ethertype))
        self.sock.bind((self.iface, 0))   # needs root / CAP_NET_RAW

        self.seq = 0
        self.frame_count = 0
        self.total_samples = 0
        self.peak = 0.0
        self.next_t = None

        print(
            "\n============================================\n"
            " ZC702 RAW ETHERNET LMS TX (float, NEON on board)\n"
            "============================================\n"
            f" Interface        : {self.iface}  (MTU {mtu})\n"
            f" PC MAC           : {self.mac_to_string(self.src_mac)}\n"
            f" ZC702 MAC        : {self.dest_mac_str}\n"
            f" EtherType        : 0x{self.ethertype:04X}\n"
            f" Samples / frame  : {self.n}\n"
            f" Frame bytes      : {30 + 16 * self.n}\n"
            f" mu (Q1.15)       : {self.mu_q15}\n"
            f" Frame gap        : {self.gap * 1e6:.0f} us\n"
            "============================================",
            flush=True,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def get_interface_mac(iface):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            info = fcntl.ioctl(s.fileno(), 0x8927,            # SIOCGIFHWADDR
                               struct.pack('256s', iface[:15].encode()))
            return info[18:24]
        finally:
            s.close()

    @staticmethod
    def get_interface_mtu(iface):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            info = fcntl.ioctl(s.fileno(), 0x8921,            # SIOCGIFMTU
                               struct.pack('256s', iface[:15].encode()))
            return struct.unpack('I', info[16:20])[0]
        finally:
            s.close()

    @staticmethod
    def mac_to_string(mac):
        return ':'.join(f'{b:02x}' for b in mac)

    # ------------------------------------------------------------------
    def build_frame(self, seq, x, d):
        """Ethernet header + 16 B app header + x[N] + d[N] (all complex64)."""
        app = struct.pack("<IIII", seq & 0xFFFFFFFF, len(x), self.mu_q15, 0)
        return (self.eth_header + app
                + np.ascontiguousarray(x, dtype=np.complex64).tobytes()
                + np.ascontiguousarray(d, dtype=np.complex64).tobytes())

    def _pace(self):
        if self.gap <= 0.0:
            return
        now = time.perf_counter()
        if self.next_t is None:
            self.next_t = now
        rem = self.next_t - now
        if rem > 300e-6:
            time.sleep(rem - 200e-6)
        while time.perf_counter() < self.next_t:
            pass
        # Schedule the NEXT frame relative to NOW, not to the old deadline.
        # With "+= gap" the deadline stays far in the past after the idle time
        # between Throttle bursts, so a whole burst went out with no spacing.
        self.next_t = time.perf_counter() + self.gap

    # ------------------------------------------------------------------
    def work(self, input_items, output_items):
        x_in = input_items[0]
        d_in = input_items[1]

        n = self.n
        frames = min(len(x_in), len(d_in)) // n
        if frames == 0:
            return 0

        for f in range(frames):
            s = f * n
            x = x_in[s:s + n]
            d = d_in[s:s + n]

            pk = max(np.abs(x.real).max(), np.abs(x.imag).max(),
                     np.abs(d.real).max(), np.abs(d.imag).max())
            if pk > self.peak:
                self.peak = float(pk)

            frame = self.build_frame(self.seq, x, d)

            self._pace()
            try:
                self.sock.send(frame)
            except Exception as ex:
                print(f"[ERROR] Ethernet send failed: {ex}", flush=True)
                return 0

            self.seq += 1
            self.frame_count += 1
            self.total_samples += n

            if self.frame_count == 1:
                print(f"[RAW ETH] first frame sent: {len(frame)} B, N={n}",
                      flush=True)
            elif self.frame_count % 500 == 0:
                print(f"[RAW ETH] frames={self.frame_count} "
                      f"samples={self.total_samples}", flush=True)

        return frames * n

    # ------------------------------------------------------------------
    def stop(self):
        try:
            self.sock.close()
        except Exception:
            pass
        msg = (f"[RAW ETH] Socket closed. Frames sent={self.frame_count}, "
               f"samples={self.total_samples}, peak |re/im|={self.peak:.3f}")
        if self.peak >= 1.0:
            msg += "  <-- exceeds 1.0: the board will saturate these samples"
        print(msg, flush=True)
        return True

        