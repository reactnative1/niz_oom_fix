import torch
import comfy.model_management as mm
from nodes import VAEDecodeTiled

def niz_log(stage):
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    free_comfy = mm.get_free_memory() / 1024**3
    stats = torch.cuda.memory_stats()
    inactive = stats.get("inactive_split.all.current", 0) / 1024**3

    print(f"\nnizzz_oom_fix [{stage}]")
    print(f"nizzz_oom_fix Allocated={alloc:.2f}GB Reserved={reserved:.2f}GB MaxAlloc={max_alloc:.2f}GB")
    print(f"nizzz_oom_fix InactiveSplit(fragmentation)={inactive:.2f}GB ComfyFree={free_comfy:.2f}GB")
    print("nizzz_oom_fix ------------------------------------------------")

class NizOOMSafeVAEDecodeTiled(VAEDecodeTiled):
    @classmethod
    def NAME(cls):
        return "VAE Decode (Tiled) [niz_oom_fix]"

    def decode_tiled(self, samples, vae, tile_size, overlap, temporal_size, temporal_overlap):
        if temporal_size > 32:
            print(f"nizzz_oom_fix clamping temporal_size {temporal_size} -> 8")
            temporal_size = 8

        niz_log("BEFORE_DECODE")
        try:
            result = super().decode_tiled(samples, vae, tile_size, overlap, temporal_size, temporal_overlap)
            niz_log("AFTER_SUCCESS")
            return result
        except torch.cuda.OutOfMemoryError as e:
            print(f"nizzz_oom_fix CAUGHT OOM: {e}")
            niz_log("AT_OOM_BEFORE_UNLOAD")
            
            mm.unload_all_models()
            niz_log("AFTER_unload_all_models")

            mm.soft_empty_cache()
            torch.cuda.empty_cache()
            niz_log("AFTER_empty_cache")

            print("nizzz_oom_fix retrying decode...")
            result = super().decode_tiled(samples, vae, tile_size, overlap, temporal_size, temporal_overlap)
            niz_log("AFTER_RETRY_SUCCESS")
            return result

NODE_CLASS_MAPPINGS = {
    "VAEDecodeTiled": NizOOMSafeVAEDecodeTiled
}