(load "package://fetcheus/fetch-interface.l")
(ros::load-ros-manifest "decopin_hand")

(fetch-init)

(defun volume-cb (msg)
  (setq *in-action* (send msg :in_action))
  (format t "*in-action*: ~A~%" *in-action*))

(defun reset-pose (&optional (tm 5000))
  (send *fetch* :reset-pose)
  (send *fetch* :rarm :move-end-pos #f(0 -100 0) :world :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) tm)
  (send *ri* :wait-interpolation))

(defun set-pose (&optional (tm 5000))
  (send *fetch* :reset-pose)
  (send *fetch* :inverse-kinematics
        (make-coords :pos #f(286 100 1030) :rpy #f(-2.674 -0.99 0.927))
        :use-torso nil)
  (send *fetch* :head :neck-y :joint-angle 89.95)
  (send *ri* :angle-vector (send *fetch* :angle-vector) 1000 :head-controller)
  (send *ri* :wait-interpolation)
  (send *ri* :angle-vector (send *fetch* :angle-vector) tm)
  (format t "Set pose ...~%")
  (send *ri* :wait-interpolation)
  (format t "Set pose done~%"))

(defun main ()
  ;; Please $ roslaunch decopin_hand audio_to_spectrogram.launch
  (setq *in-action* nil)
  ;; (ros::get-param "/action_detector_volume/power_per_pixel_threshold")
  ;; (ros::set-dynamic-reconfigure-param
  ;;  "/action_detector_volume"
  ;;  "power_per_pixel_threshold" :double 0.35)
  ;; (ros::subscribe "/action_detector_volume/in_action"
  ;;                 decopin_hand::InAction #'volume-cb)
  (set-pose)
  ;; Collecting hammering vibrations
  (format t "Start hammering~%")
  (dotimes (i 1)
    (ros::spin-once)
    ;; Hammering
    (let ((diff 25))
      (send *fetch* :head :neck-y :joint-angle
            (- (send *fetch* :head :neck-y :joint-angle ) diff))
      (send *ri* :angle-vector (send *fetch* :angle-vector) 1000)
      (send *ri* :wait-interpolation)
      ;; Reset
      (send *fetch* :head :neck-y :joint-angle
            (- (send *fetch* :head :neck-y :joint-angle ) (* -1 diff)))
      (send *ri* :angle-vector (send *fetch* :angle-vector) 1000)
      (send *ri* :wait-interpolation)))
  )
