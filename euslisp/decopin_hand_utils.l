(require "package://decopin_hand/euslisp/decopin_hand_interface.l")

(defmethod decopin_hand-robot
  (:inverse-kinematics
   (target-coords &rest args &key link-list move-arm move-target &allow-other-keys)
   (unless move-target (setq move-target (send self :rarm :end-coords)))
   (unless link-list (setq link-list (send self :link-list (send move-target :parent))))
   (send-super* :inverse-kinematics target-coords
                :move-target move-target
                :link-list link-list
                args))
  )

(defun ik-example ()
  (send *robot* :reset-pose)
  (send *robot* :inverse-kinematics (make-coords :pos #f(-30 100 150)) :rotation-axis nil :revert-if-fail nil)
  (send *robot* :rarm :end-coords))


;; Close gripper: (+ 0 + -)
;; Open gripper : (- 0 - +)
(defun move-gripper (diff-av &optional (tm 3000) &key (wait t))
  (send *ri* :angle-vector
        (send *robot* :angle-vector
              (v+ (send *robot* :angle-vector) diff-av))
        tm)
  (if wait
    (send *ri* :wait-interpolation))
  )

(ros::load-ros-manifest "jsk_recognition_msgs")
;; The *continuous-class* is output if the same *temporary-class* continues for *duration-thre* [s]
(setq *duration-thre* 0.4)
(setq *temporary-class* nil)
(setq *continuous-class* nil)
(defun action-cb (msg)
  ;; (setq *msg-debug* msg) ;; for debugging
  (setq *temporary-class* (string-right-trim '(#\Newline) (car (send msg :label_names)))))
(ros::subscribe "/action_classifier/output" jsk_recognition_msgs::ClassificationResult #'action-cb)

;; Publish the gesture which the robot hand recognize
(ros::advertise "/gesture" jsk_recognition_msgs::ClassificationResult 1)
(defun recognize-gesture (continuous-class) continuous-class) ;; Please overwrite
(setq *class-to-label* (make-hash-table :test #'equal))
(defun stable-action-cb (msg)
  ;; (setq *stable-msg-debug* msg) ;; for debugging
  (let* ((msg-copy (copy-object msg))
        (gesture (recognize-gesture *continuous-class*))
        (cls (string-right-trim '(#\Newline) (car (send msg-copy :label_names)))))
    (when (null (gethash cls *class-to-label*))
      (sethash cls *class-to-label* (send msg-copy :labels)))
    (setq *continuous-class* cls)
    (if (equal gesture "dead_time")
      (send msg-copy :labels (list 0)) ;; Set label color as black
      (send msg-copy :labels (gethash gesture *class-to-label*)))
    (send msg-copy :label_names (list (format nil "~A~%" gesture)))
    ;; (format t "labels: ~A, label_names; ~A" (send msg-copy :labels) (send msg-copy :label_names))
    (ros::publish "/gesture" msg-copy))
    )
(ros::subscribe "/continuous_filter/output" jsk_recognition_msgs::ClassificationResult #'stable-action-cb)

;; Please $ roslaunch decopin_hand audio_to_spectrogram.launch
(setq *in-action* nil)
;; (ros::set-dynamic-reconfigure-param
;;  "/action_detector_volume"
;;  "power_per_pixel_threshold" :double 0.35)
(ros::load-ros-manifest "decopin_hand")
(defun volume-cb (msg)
  (setq *in-action* (send msg :in_action))
  (format t "*in-action*: ~A~%" *in-action*))
(ros::subscribe "/action_detector_volume/in_action"
                decopin_hand::InAction #'volume-cb)

(defun print-state ()
  (format t "*temporary-class*: ~A~%" *temporary-class*)
  (format t "*continuous-class*: ~A~%~%" *continuous-class*))
