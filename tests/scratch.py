from src.io.volume_reader import VolumeReader

vr = VolumeReader()
vr.create_volume_list()

print(vr.normalization_constants)

print(vr.normalization_constants['mean'])

