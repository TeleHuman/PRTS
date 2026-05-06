import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import torch
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        delta2abs = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self._delta2abs = delta2abs

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()

                device = 'cuda'

                def _to_bfloat16_device_tensor(np_array, device, is_image=False):
                    # tensor = torch.from_numpy(np_array.transpose(2, 0, 1).astype(np.float32))
                    if is_image:
                        tensor = torch.from_numpy(np_array.transpose(2, 0, 1).astype(np.float32)) / 255.
                    else:
                        tensor = torch.from_numpy(np_array.astype(np.float32))
                    tensor = tensor.unsqueeze(0)
                    return tensor.to(device).to(torch.float32) # .to(torch.bfloat16)

                # Dynamically process all observation keys
                lerobot_format_obs = {}
                for key, value in obs.items():  
                    if key == "observation/state":  
                        # Handle state separately (not an image)  
                        lerobot_format_obs["observation.state"] = _to_bfloat16_device_tensor(  
                            value, device, is_image=False  
                        )
                    elif key.startswith("observation/"):  
                        # Only treat as image if there is exactly one segment after "observation/"  
                        remainder = key[len("observation/"):]  
                        if remainder and "/" not in remainder:  
                            image_key = f"observation.images.{remainder}"  
                            lerobot_format_obs[image_key] = _to_bfloat16_device_tensor(  
                                value, device, is_image=True  
                            )
                    elif key == "prompt":
                        # Handle prompt
                        lerobot_format_obs["task"] = [value]

                # lerobot_format_obs = {
                #     "observation.images.image": _to_bfloat16_device_tensor(obs["observation/image"], device, is_image=True),
                #     "observation.images.wrist_image": _to_bfloat16_device_tensor(obs["observation/wrist_image"], device, is_image=True),
                #     "observation.state": _to_bfloat16_device_tensor(obs["observation/state"], device, is_image=False),
                #     "task": [obs["prompt"]],
                # }

                action = self._policy.infer(lerobot_format_obs)
                
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
