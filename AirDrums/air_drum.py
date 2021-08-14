# organize imports
import cv2
import numpy as np
from pygame import mixer

# color to detect - drum stick
lower_red = [0, 177, 212]
upper_red = [179, 255, 255]
lower_blue = [93, 124, 145]
upper_blue = [137, 255, 241]

# initialize mixer
mixer.init()

# region coordinates
k_top, k_bottom, k_right, k_left = 580, 720, 640, 940
h_top, h_bottom, h_right, h_left = 200, 300, 340, 540
s_top, s_bottom, s_right, s_left = 300, 400, 540, 740
c_top, c_bottom, c_right, c_left = 60, 160, 880, 1080
ch_top, ch_bottom, ch_right, ch_left = 580, 720, 335, 635
ht_top, ht_bottom, ht_right, ht_left = 50, 150, 400, 600
mt_top, mt_bottom, mt_right, mt_left = 50, 150, 620, 820
ft_top, ft_bottom, ft_right, ft_left = 300, 400, 840, 1040

#----------------------
# play sounds
#----------------------
def playKick():
	mixer.music.load('kick.mp3')
	mixer.music.play()

def playHihat():
	mixer.music.load('hihat.mp3')
	mixer.music.play()

def playSnare():
	mixer.music.load('snare.mp3')
	mixer.music.play()

def playCrash():
	mixer.music.load('crash.mp3')
	mixer.music.play()

def playClosed():
	mixer.music.load('closed.mp3')
	mixer.music.play()

def playHitom():
	mixer.music.load('hitom.mp3')
	mixer.music.play()

def playMidtom():
	mixer.music.load('midtom.mp3')
	mixer.music.play()

def playFloortom():
	mixer.music.load('floortom.mp3')
	mixer.music.play()

#----------------------
# find contours
#----------------------
def findContours(image):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresholded = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)[1]
	(cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return len(cnts)
	

# bool for each drum
e_snare = 0
e_kick  = 0
e_hihat = 0
e_crash = 0
e_closed = 0
e_hitom = 0
e_mitom = 0
e_ftom = 0

#----------------------
# main function
#----------------------
if __name__ == "__main__":
	# accumulated weight
	aWeight = 0.5

	# get reference to camera
	cam = cv2.VideoCapture(0)

	# camera related tuning
	cam.set(3, 1280)
	cam.set(4, 720)
	cam.set(cv2.CAP_PROP_FPS, 60)

	bg=cv2.imread('bg32.png')
	bg_height, bg_width, bg_c = bg.shape
	bgm=cv2.resize(bg, (1280,720))
	# bgm=cv2.flip(bgm,1)
	x=0
	y=0

	# loop till user presses "q"
	while True:
		# read a frame from the camera
		status, frame = cam.read()
		
		#loading background
		#frame[ y:y+bg_height , x:x+bg_width ] = bgm

		# take a clone 
		clone = frame.copy()
		clone = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		clone = cv2.flip(clone, 1)
		clone = cv2.resize(clone, (1280,720))
				
		# get the three drum regions
		reg_kick  = clone[k_top:k_bottom, k_right:k_left]
		reg_hihat = clone[h_top:h_bottom, h_right:h_left]
		reg_snare = clone[s_top:s_bottom, s_right:s_left]
		reg_crash = clone[c_top:c_bottom, c_right:c_left]
		reg_closed = clone[ch_top:ch_bottom, ch_right:ch_left]
		reg_hitom = clone[ht_top:ht_bottom, ht_right:ht_left]
		reg_mitom = clone[mt_top:mt_bottom, mt_right:mt_left]
		reg_ftom = clone[ft_top:ft_bottom, ft_right:ft_left]

		# blur the regions
		reg_kick  = cv2.GaussianBlur(reg_kick,  (7, 7), 0)
		reg_hihat = cv2.GaussianBlur(reg_hihat, (7, 7), 0)
		reg_snare = cv2.GaussianBlur(reg_snare, (7, 7), 0)
		reg_crash = cv2.GaussianBlur(reg_crash, (7, 7), 0)
		reg_closed = cv2.GaussianBlur(reg_closed, (7, 7), 0)
		reg_hitom = cv2.GaussianBlur(reg_hitom, (7, 7), 0)
		reg_mitom = cv2.GaussianBlur(reg_mitom, (7, 7), 0)
		reg_ftom = cv2.GaussianBlur(reg_ftom, (7, 7), 0)

		l = np.array(lower_red, dtype="uint8")
		u = np.array(upper_red, dtype="uint8")
		lb = np.array(lower_blue, dtype="uint8")
		ub = np.array(upper_blue, dtype="uint8")
		
		mask_kick  = cv2.inRange(reg_kick,  l, u)
		mask_hihat = cv2.inRange(reg_hihat, lb, ub)
		mask_snare = cv2.inRange(reg_snare, l, u)
		mask_crash = cv2.inRange(reg_crash, l, u)
		mask_closed = cv2.inRange(reg_closed, l, u)
		mask_hitom = cv2.inRange(reg_hitom, l, u)
		mask_mitom = cv2.inRange(reg_mitom, l, u)
		mask_ftom = cv2.inRange(reg_ftom, l, u)

		out_kick   = cv2.bitwise_and(reg_kick,  reg_kick,  mask=mask_kick)
		out_hihat  = cv2.bitwise_and(reg_hihat, reg_hihat, mask=mask_hihat)
		out_snare  = cv2.bitwise_and(reg_snare, reg_snare, mask=mask_snare)
		out_crash  = cv2.bitwise_and(reg_crash, reg_crash, mask=mask_crash)
		out_closed  = cv2.bitwise_and(reg_closed, reg_closed, mask=mask_closed)
		out_hitom  = cv2.bitwise_and(reg_hitom, reg_hitom, mask=mask_hitom)
		out_mitom  = cv2.bitwise_and(reg_mitom, reg_mitom, mask=mask_mitom)
		out_ftom  = cv2.bitwise_and(reg_ftom, reg_ftom, mask=mask_ftom)

		cnts_kick  = findContours(out_kick)
		cnts_hihat = findContours(out_hihat)
		cnts_snare = findContours(out_snare)
		cnts_crash = findContours(out_crash)
		cnts_closed = findContours(out_closed)
		cnts_hitom = findContours(out_hitom)
		cnts_mitom = findContours(out_mitom)
		cnts_ftom = findContours(out_ftom)

		# drawing box around red stick
		red_stick = cv2.inRange(clone, l, u)
		blue_stick = cv2.inRange(clone, lb, ub)

		sticks,hierachy=cv2.findContours(red_stick,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		sticks = sorted(sticks, key=lambda x:cv2.contourArea(x), reverse=True)

    	#startpoint, endpoint, color, thickness
		for box in sticks:
			(x,y,w,h) = cv2.boundingRect(box)
			cv2.rectangle(clone,(x,y),(x + w, y + h),(0,255,0),2) 
			# cv2.putText(clone, 'LH',(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,255), 1, cv2.LINE_AA)

		# drawing box around blue stick
		sticks,hierachy=cv2.findContours(blue_stick,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		sticks = sorted(sticks, key=lambda x:cv2.contourArea(x), reverse=True)

    	#startpoint, endpoint, color, thickness
		for box in sticks:
			(x,y,w,h) = cv2.boundingRect(box)
			cv2.rectangle(clone,(x,y),(x + w, y + h),(0,255,0),2)
			# cv2.putText(clone, 'RH',(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,0,0), 1, cv2.LINE_AA)
			
		# detecting stick and playing
		# only kick
		if (cnts_kick > 0) and (e_kick == 0):
			playKick()
			e_kick = 1
		elif (cnts_kick == 0):
		 	e_kick = 0

		# Kick and snare
		if (cnts_kick > 0) and (e_kick == 0) and (e_snare > 0) and (e_snare == 0):
			playKick()
			playSnare()
			e_kick = 1
			e_snare = 1
		elif (cnts_kick == 0) and (cnts_snare == 0):
			e_kick = 0
			e_snare = 0

		# Kick and hihat
		if (cnts_kick > 0) and (e_kick == 0) and (e_hihat > 0) and (e_hihat == 0):
			playKick()
			playHihat()
			e_kick = 1
			e_hihat = 1
		elif (cnts_kick == 0) and (cnts_hihat == 0):
			e_kick = 0
			e_hihat = 0

		# kick and crash
		if (cnts_kick > 0) and (e_kick == 0) and (e_crash > 0) and (e_crash== 0):
			playKick()
			playCrash()
			e_kick = 1
			e_crash = 1
		elif (cnts_kick == 0) and (cnts_crash == 0):
			e_kick = 0
			e_crash = 0

		# Onlysnare
		if (cnts_snare > 0) and (e_snare == 0):
			playSnare()
			e_snare = 1
		elif (cnts_snare == 0):
			e_snare = 0

		# Snare and crash
		if (cnts_snare > 0) and (e_snare == 0) and (e_crash > 0) and (e_crash == 0):
			playSnare()
			playCrash()
			e_snare = 1
			e_crash = 1
		elif (cnts_snare == 0) and (cnts_crash == 0):
			e_snare = 0
			e_crash = 0

		# Snare and hihat
		if (cnts_snare > 0) and (e_snare == 0) and (cnts_hihat > 0) and (e_hihat == 0):
			playSnare()
			playHihat()
			e_snare = 1
			e_hihat = 1
		elif (cnts_snare == 0) and (cnts_hihat == 0):
			e_snare = 0
			e_hihat = 0
			
		# Hihat and Crash
		if (cnts_hihat > 0) and (e_hihat == 0) and (e_crash > 0) and (e_crash == 0):
			playHihat()
			playCrash()
			e_hihat = 1
			e_crash = 1
		elif (cnts_hihat == 0) and (cnts_crash == 0):
			e_hihat = 0
			e_crash = 0

		# Onlyclosedhihat
		if (cnts_closed > 0) and (cnts_hihat > 0) and (e_hihat == 0) and (e_closed == 0):
			playClosed()
			e_hihat = 1
			e_closed = 1
		elif (cnts_hihat == 0) and (cnts_closed > 0):
			e_hihat = 0
			e_closed = 0
		elif (cnts_closed > 0) and (cnts_hihat > 0) and (e_hihat == 0) and (e_closed == 0):
			playClosed()
			e_hihat = 1
			e_closed = 1	
		elif (cnts_hihat == 0) and (cnts_closed == 0):
			e_hihat = 0
			e_closed = 0

		# Onlyopenhihat
		if (cnts_hihat > 0) and (e_hihat == 0):
			playHihat()
			e_hihat = 1
		elif (cnts_hihat == 0):
			e_hihat = 0

		# Onlycrash
		if (cnts_crash > 0) and (e_crash == 0):
			playCrash()
			e_crash = 1
		elif (cnts_crash == 0):
			e_crash = 0
		
		# OnlyHitom
		if (cnts_hitom > 0) and (e_hitom == 0):
			playHitom()
			e_hitom = 1
		elif (cnts_hitom == 0):
		 	e_hitom = 0
		
		# OnlyMidtom
		if (cnts_mitom > 0) and (e_mitom == 0):
			playMidtom()
			e_mitom = 1
		elif (cnts_mitom == 0):
		 	e_mitom = 0
		
		# OnlyFloortom
		if (cnts_ftom > 0) and (e_ftom == 0):
			playFloortom()
			e_ftom = 1
		elif (cnts_ftom == 0):
		 	e_ftom = 0

		# draw the drum regions
		# cv2.rectangle(clone, (k_left,k_top), (k_right,k_bottom), (0,255,0,0.5), 2)
		# cv2.rectangle(clone, (h_left,h_top), (h_right,h_bottom), (255,0,0,0.5), 2)
		# cv2.rectangle(clone, (s_left,s_top), (s_right,s_bottom), (0,0,255,0.5), 2)
		# cv2.rectangle(clone, (c_left,c_top), (c_right,c_bottom), (0,0,255,0.5), 2)
		# cv2.rectangle(clone, (ch_left,ch_top), (ch_right,ch_bottom), (255,0,0,0.5), 2)
		# cv2.rectangle(clone, (ht_left,ht_top), (ht_right,ht_bottom), (255,0,0,0.5), 2)
		# cv2.rectangle(clone, (mt_left,mt_top), (mt_right,mt_bottom), (255,0,0,0.5), 2)
		# cv2.rectangle(clone, (ft_left,ft_top), (ft_right,ft_bottom), (255,0,0,0.5), 2)
	
		# display the frame
		cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
		norm = cv2.cvtColor(clone, cv2.COLOR_HSV2BGR)
		norm = norm + bgm
		cv2.imshow("video", norm)
		# cv2.imshow("mask", out_hihat )
		# cv2.imshow("snare", out_snare)
		# cv2.imshow("kick", out_kick)
		# cv2.imshow("closed", out_closed)

		# if user presses 'q', quit the program
		if cv2.waitKey(1) & 0XFF == ord('q'):
			break

	# release the camera
	cam.release()

	# destroy all windows
	cv2.destroyAllWindows()