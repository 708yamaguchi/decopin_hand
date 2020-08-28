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
(setq *last-time* nil)
(setq *duration-thre* 0.4)
(setq *temporary-class* nil)
(setq *continuous-class* nil)
(setq *gesture-accepting* t)
(defun action-cb (msg)
  ;; (setq *msg-debug* msg) ;; for debugging
  (setq *temporary-class* (string-right-trim '(#\Newline) (car (send msg :label_names)))))
(ros::subscribe "/action_classifier/output" jsk_recognition_msgs::ClassificationResult #'action-cb)

;; Publish the gesture which the robot hand recognize
(ros::advertise "/gesture" jsk_recognition_msgs::ClassificationResult 1)
(defun stable-action-cb (msg)
  ;; (setq *stable-msg-debug* msg) ;; for debugging
  (setq *continuous-class* (string-right-trim '(#\Newline) (car (send msg :label_names))))
  (when (not *gesture-accepting*)
    (send msg :labels (list 0)) ;; Set label color as black
    (send msg :label_names (list (format nil "dead_time~%")))
    (if (null *last-time*)
      (setq *last-time* (send (ros::time-now) :to-sec)))
    (setq *continuous-class* "dead_time"))
  (ros::publish "/gesture" msg)
  )
(ros::subscribe "/continuous_filter/output" jsk_recognition_msgs::ClassificationResult #'stable-action-cb)


(defun print-state ()
  (format t "*temporary-class*: ~A~%" *temporary-class*)
  (format t "*continuous-class*: ~A~%~%" *continuous-class*))
