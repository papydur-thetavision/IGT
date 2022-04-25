from src.io.volume_reader import VolumeReader

vr = VolumeReader()
vr.create_volume_list()


vr.precompute_endpoints()

end = 0
