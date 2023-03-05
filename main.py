import moviepy.editor

video = moviepy.editor.VideoFileClip('ridoy.mp4')

audio = video.audio

audio.write_audiofile("ridoy.wav")