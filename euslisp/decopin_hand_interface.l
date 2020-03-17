(ros::roseus "decopin_hand")

;; Before using this script, please create euslisp model
;; based on README.md

(require "package://pr2eus/robot-interface.l")
(require "package://decopin_hand/euslisp/decopin_hand.l")
(require "package://euscollada/src/euscollada-robot.l")

(defclass decopin-hand-interface
  :super robot-interface
  :slots (hoge)
  )

(defmethod decopin-hand-interface
  (:init (&rest args)
   (prog1 (send-super* :init :robot decopin_hand-robot :base-controller-action-name nil :joint-states-topic "dynamixel_workbench/joint_states" args)
   (send self :add-controller :fullbody-controller)
   ))
  (:default-controller
   ()
   (append
    (send self :fullbody-controller))
   )
  (:fullbody-controller
   ()
   (list
    (list
     (cons :controller-action "dynamixel_workbench/follow_joint_trajectory_action")
     (cons :controller-state "dynamixel_workbench/follow_joint_trajectory_action/state")
     (cons :action-type control_msgs::FollowJointTrajectoryAction)
     (cons :joint-names (send-all (send robot :joint-list) :name))))
   )
  (:angle-vector
   (av &optional (tm nil) (ctype controller-type) (start-time 0) &key (min-time 0.01) &rest args)
   (send-super* :angle-vector av tm ctype start-time :min-time min-time args)
   )
  )

(defun decopin-hand-init ()
  (unless (boundp '*robot*)
    (setq *robot* (instance decopin_hand-robot :init)))
  (unless (boundp '*ri*)
    (setq *ri* (instance decopin-hand-interface :init))))
