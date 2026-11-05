"""Serializable observation schema shared by the experiment plan and workers."""

OBJECT_CHANNELS = ("Player", "Shield", "Bullet", "Satellite", "Alien")
OBJECT_CHANNEL_SCREEN = (160, 210)  # width, height


def object_channel_spec(width, height, stack_size):
    return {
        "version": 1,
        "channels": list(OBJECT_CHANNELS),
        "source_size": list(OBJECT_CHANNEL_SCREEN),
        "map_size": [width, height],
        "stack_size": stack_size,
        "stack_order": "oldest_first",
        "rasterization": "binary_union_clip_floor_start_ceil_end",
    }
