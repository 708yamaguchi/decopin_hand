(load "package://fetcheus/fetch-interface.l")
(ros::load-ros-manifest "decopin_hand")

(fetch-init)

(defun volume-cb (msg)
  (setq *in-action* (send msg :in_action))
  (format t "*in-action*: ~A~%" *in-action*))

(defun reset-pose ()
  (send *fetch* :reset-pose)
  (send *fetch* :rarm :move-end-pos #f(0 -100 0) :world :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
  (send *ri* :wait-interpolation))

(defun set-pose (&optional (tm 5000))
  (send *fetch* :reset-pose)
  (send *fetch* :torso :waist-z :joint-angle 200)
  (send *fetch* :inverse-kinematics
        (make-coords :pos #f(700 0 850) :rpy (float-vector 0 0 -2.0))
        :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) tm)
  (format t "Set pose ...~%")
  (send *ri* :wait-interpolation)
  (format t "Set pose done~%"))

(defun main ()
  ;; Please $ roslaunch decopin_hand audio_to_spectrogram.launch
  (setq *in-action* nil)
  ;; (ros::get-param "/action_detector_volume/power_per_pixel_threshold")
  (ros::set-dynamic-reconfigure-param
   "/action_detector_volume"
   "power_per_pixel_threshold" :double 0.35)
  (ros::subscribe "/action_detector_volume/in_action"
                  decopin_hand::InAction #'volume-cb)
  (set-pose)
  ;; Collecting hammering vibrations
  (format t "Start hammering~%")
  (dotimes (i 3)
    (ros::spin-once)
    (send *fetch* :rarm :move-end-pos #f(0 0 -50) :world :use-torso nil)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 2000)
    ;; Wait for :interpolatingp becomes t
    (while (null (send *ri* :interpolatingp))
      (unix:usleep (* 100 1000)))
    ;; During robot movement, stop robot motion when *in-action* becomes t
    (while (and (send *ri* :interpolatingp)
                (null *in-action*))
      (ros::spin-once)
      (unix:usleep (* 10 1000)))
    (when *in-action*
      (format t "action detected. :cancel-angle-vector~%")
      (send *ri* :cancel-angle-vector))
    ;; Reset to (set-pose)
    (set-pose 2000)
  )
