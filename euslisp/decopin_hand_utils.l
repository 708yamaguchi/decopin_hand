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
;; The *stable-class* is output if the same *temporary-class* continues for *duration-thre* [s]
(setq *last-time* nil)
(setq *duration-thre* 0.4)
(setq *temporary-class* nil)
(setq *stable-class* nil)
(defun action-cb (msg)
  ;; (setq *msg-debug* msg) ;; for debugging
  (let* ((label-name (string-right-trim '(#\Newline) (car (send msg :label_names))))
         (header-stamp (send msg :header :stamp))
         (now-time (+ (send (ros::time-now) :sec) (* (send (ros::time-now) :nsec) 0.000000001))))
    ;; Check if temporary class changes
    (when (not (equal label-name *temporary-class*))
      (setq *stable-class* nil)
      (setq *last-time* now-time))
    (setq *temporary-class* label-name)
    ;; Check if the same *temporary-class* continues for *duration-thre* [s], for debugging
    ;; (format t "*last-time*: ~9,9F~%now-time: ~9,9F~%" *last-time* now-time)
    (when (> (- now-time *last-time*)
             *duration-thre*)
      (when (not (equal *stable-class* *temporary-class*))
        (format t "*stable-class* changed from ~A to ~A~%"
                *stable-class* *temporary-class*))
      (setq *stable-class* *temporary-class*))))
(ros::subscribe "/action_classifier/output" jsk_recognition_msgs::ClassificationResult #'action-cb)

(defun print-state ()
  (format t "*temporary-class*: ~A~%" *temporary-class*)
  (format t "*stable-class*: ~A~%~%" *stable-class*))
