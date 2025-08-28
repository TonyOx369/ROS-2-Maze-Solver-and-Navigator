#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <map>
#include <set>

// Define the A* Node structure
struct AStarNode {
    int x, y; // Grid coordinates
    double g_cost = 0.0; // Cost from start to current node
    double h_cost = 0.0; // Heuristic (estimated cost from current to end)
    AStarNode* parent = nullptr;
    double f_cost() const { return g_cost + h_cost; }
};

class MazeSolverNode : public rclcpp::Node
{
public:
    MazeSolverNode();
    ~MazeSolverNode(); 

private:
    // Callback functions for subscribers
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void entryCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void exitCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

    // Main logic functions
    void startNavigation();
    void inflateMap();
    void aStarSearch();
    void smoothPath();
    void controlLoop();
    double getYawFromQuaternion(const geometry_msgs::msg::Quaternion& q);
    bool isValidCell(int x, int y, const nav_msgs::msg::OccupancyGrid& map);
    double calculateDistance(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2);

    // Custom comparator for the A* priority queue (stores pointers)
    struct AStarNodePointerCompare {
        bool operator()(const AStarNode* a, const AStarNode* b) const {
            return a->f_cost() > b->f_cost();
        }
    };

    // ROS 2 publishers, subscribers, and timers
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr entry_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr exit_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr planned_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr smoothed_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr actual_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr inflated_map_pub_;
    rclcpp::TimerBase::SharedPtr controller_timer_;

    // Member variables to store state
    nav_msgs::msg::OccupancyGrid current_map_;
    nav_msgs::msg::OccupancyGrid inflated_map_;
    geometry_msgs::msg::PoseStamped entry_pose_;
    geometry_msgs::msg::PoseStamped exit_pose_;
    geometry_msgs::msg::Pose current_pose_;
    nav_msgs::msg::Path planned_path_;
    nav_msgs::msg::Path smoothed_path_;
    nav_msgs::msg::Path actual_path_;
    size_t current_waypoint_index_ = 0;
    bool map_received_ = false;
    bool entry_received_ = false;
    bool exit_received_ = false;
    bool path_planned_ = false;

    // Control parameters
    double kp_linear_ = 0.25;
    double kp_angular_ = 0.4;
    double max_linear_vel_ = 0.15;
    double max_angular_vel_ = 0.3;
    double goal_radius_ = 0.25;
    double inflation_radius_ = 0.3; // meters
    double heading_tolerance_ = 0.3; // radians
    double lookahead_distance_ = 0.4; // meters for path following

    // A* memory management
    std::vector<AStarNode*> a_star_nodes_;
};


MazeSolverNode::MazeSolverNode() : Node("maze_solver_node")
{
    using std::placeholders::_1;

    // QoS profile for subscribers that receive data once
    rclcpp::QoS qos_subscriber_profile(
        rclcpp::QoSInitialization(
            RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            10
        ));
    qos_subscriber_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_subscriber_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    // "Latching" QoS profile for publishers of static data (maps, paths)
    rclcpp::QoS qos_latching_publisher_profile(
        rclcpp::QoSInitialization(
            RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            1
        ));
    qos_latching_publisher_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_latching_publisher_profile.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);


    // Subscribers
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/virya_test/map", qos_subscriber_profile, std::bind(&MazeSolverNode::mapCallback, this, _1));

    entry_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/virya_test/entry", qos_subscriber_profile, std::bind(&MazeSolverNode::entryCallback, this, _1));

    exit_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/virya_test/exit", qos_subscriber_profile, std::bind(&MazeSolverNode::exitCallback, this, _1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/virya_test/odom", 10, std::bind(&MazeSolverNode::odomCallback, this, _1));

    // Publishers
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 10);
    actual_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/actual_path", 10);

    inflated_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/inflated_map", qos_latching_publisher_profile);

    planned_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
        "/planned_path", qos_latching_publisher_profile);

    smoothed_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
        "/smoothed_path", qos_latching_publisher_profile);

    // Controller timer
    controller_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(50), std::bind(&MazeSolverNode::controlLoop, this));

    RCLCPP_INFO(this->get_logger(), "Maze Solver Node has started and is waiting for data.");
}

MazeSolverNode::~MazeSolverNode()
{
    for (auto node : a_star_nodes_) {
        delete node;
    }
}

// Callback functions
void MazeSolverNode::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    if (!map_received_) {
        RCLCPP_INFO(this->get_logger(), "Map received. Resolution: %.3f, Size: %dx%d",
                   msg->info.resolution, msg->info.width, msg->info.height);
        current_map_ = *msg;
        map_received_ = true;
        startNavigation();
    }
}

void MazeSolverNode::entryCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (!entry_received_) {
        RCLCPP_INFO(this->get_logger(), "Entry point received at (%.2f, %.2f).",
                   msg->pose.position.x, msg->pose.position.y);
        entry_pose_ = *msg;
        entry_received_ = true;
        startNavigation();
    }
}

void MazeSolverNode::exitCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (!exit_received_) {
        RCLCPP_INFO(this->get_logger(), "Exit point received at (%.2f, %.2f).",
                   msg->pose.position.x, msg->pose.position.y);
        exit_pose_ = *msg;
        exit_received_ = true;
        startNavigation();
    }
}

void MazeSolverNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    current_pose_ = msg->pose.pose;

    // For RViz visualization of the robot's actual path
    if (path_planned_) {
        geometry_msgs::msg::PoseStamped current_pose_stamped;
        current_pose_stamped.header = msg->header;
        current_pose_stamped.pose = current_pose_;
        actual_path_.header = msg->header;
        actual_path_.header.frame_id = "map";

        // Only add if we've moved significantly (reduce path density)
        if (actual_path_.poses.empty() ||
            calculateDistance(actual_path_.poses.back().pose.position, current_pose_.position) > 0.1) {
            actual_path_.poses.push_back(current_pose_stamped);
        }
        actual_path_pub_->publish(actual_path_);
    }
}

// Core Logic
void MazeSolverNode::startNavigation() {
    if (map_received_ && entry_received_ && exit_received_ && !path_planned_) {
        RCLCPP_INFO(this->get_logger(), "All data received. Starting path planning...");
        inflateMap();
        aStarSearch();
        if (path_planned_) {
            smoothPath();
        }
    }
}

void MazeSolverNode::inflateMap() {
    RCLCPP_INFO(this->get_logger(), "Inflating map for safety...");
    inflated_map_ = current_map_;
    float resolution = current_map_.info.resolution;

    // Ensure minimum inflation radius
    int inflation_radius_cells = std::max(2, static_cast<int>(std::ceil(inflation_radius_ / resolution)));

    RCLCPP_INFO(this->get_logger(), "Inflation radius: %.2f meters (%d cells)",
               inflation_radius_, inflation_radius_cells);

    int width = current_map_.info.width;
    int height = current_map_.info.height;
    std::vector<std::pair<int, int>> obstacle_cells;

    // Find all original obstacles
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            if (current_map_.data[index] > 50) // Occupied or unknown
            { 
                obstacle_cells.push_back({x, y});
            }
        }
    }

    // Inflate around each original obstacle using circular inflation
    for (const auto& obs : obstacle_cells) {
        int ox = obs.first;
        int oy = obs.second;

        for (int dy = -inflation_radius_cells; dy <= inflation_radius_cells; ++dy) 
        {
            for (int dx = -inflation_radius_cells; dx <= inflation_radius_cells; ++dx) 
            {
                // Use circular inflation instead of square
                double distance = std::sqrt(dx * dx + dy * dy);
                if (distance <= inflation_radius_cells) 
                {
                    int nx = ox + dx;
                    int ny = oy + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) 
                    {
                        inflated_map_.data[ny * width + nx] = 100;
                    }
                }
            }
        }
    }

    // Publish inflated map for visualization
    inflated_map_.header = current_map_.header;
    inflated_map_pub_->publish(inflated_map_);

    RCLCPP_INFO(this->get_logger(), "Map inflation complete.");
}

bool MazeSolverNode::isValidCell(int x, int y, const nav_msgs::msg::OccupancyGrid& map) {
    if (x < 0 || x >= (int)map.info.width || y < 0 || y >= (int)map.info.height) {
        return false;
    }
    int index = y * map.info.width + x;
    return map.data[index] <= 0; // Free space (value 0)
}

void MazeSolverNode::aStarSearch() {
    RCLCPP_INFO(this->get_logger(), "Starting A* search...");

    float resolution = inflated_map_.info.resolution;
    float origin_x = inflated_map_.info.origin.position.x;
    float origin_y = inflated_map_.info.origin.position.y;

    int start_x = static_cast<int>((entry_pose_.pose.position.x - origin_x) / resolution);
    int start_y = static_cast<int>((entry_pose_.pose.position.y - origin_y) / resolution);
    int goal_x = static_cast<int>((exit_pose_.pose.position.x - origin_x) / resolution);
    int goal_y = static_cast<int>((exit_pose_.pose.position.y - origin_y) / resolution);

    RCLCPP_INFO(this->get_logger(), "Start: (%d, %d), Goal: (%d, %d)", start_x, start_y, goal_x, goal_y);

    if (!isValidCell(start_x, start_y, inflated_map_)) {
        RCLCPP_ERROR(this->get_logger(), "Start position is in an obstacle!");
        return;
    }
    if (!isValidCell(goal_x, goal_y, inflated_map_)) {
        RCLCPP_ERROR(this->get_logger(), "Goal position is in an obstacle!");
        return;
    }

    std::priority_queue<AStarNode*, std::vector<AStarNode*>, AStarNodePointerCompare> open_list;
    std::map<int, AStarNode*> all_nodes;
    std::set<int> closed_set;

    // --- INITIALIZATION ---
    AStarNode* start_node = new AStarNode{start_x, start_y, 0.0, 0.0, nullptr};
    a_star_nodes_.push_back(start_node); 
    start_node->h_cost = std::hypot(start_x - goal_x, start_y - goal_y);

    open_list.push(start_node);
    int start_idx = start_y * inflated_map_.info.width + start_x;
    all_nodes[start_idx] = start_node;

    AStarNode* goal_node = nullptr;
    int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};
    double costs[] = {1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414}; // Cost for 8-way movement

    int iterations = 0;
    const int max_iterations = 50000;

    while (!open_list.empty() && iterations < max_iterations) {
        iterations++;

        AStarNode* current_node = open_list.top();
        open_list.pop();

        int current_idx = current_node->y * inflated_map_.info.width + current_node->x;

        if (closed_set.count(current_idx)) {
            continue;
        }
        closed_set.insert(current_idx);

        // --- GOAL CHECK ---
        if (current_node->x == goal_x && current_node->y == goal_y) {
            goal_node = current_node;
            RCLCPP_INFO(this->get_logger(), "Path found!");
            break;
        }

        // --- EXPLORE NEIGHBORS ---
        for (int i = 0; i < 8; ++i) {
            int nx = current_node->x + dx[i];
            int ny = current_node->y + dy[i];
            int neighbor_idx = ny * inflated_map_.info.width + nx;

            if (!isValidCell(nx, ny, inflated_map_) || closed_set.count(neighbor_idx)) {
                continue;
            }

            double new_g_cost = current_node->g_cost + costs[i];
            AStarNode* neighbor_node = nullptr;

            // --- CORRECT OPEN SET HANDLING ---
            auto it = all_nodes.find(neighbor_idx);
            if (it == all_nodes.end()) {
                // First time seeing this node
                neighbor_node = new AStarNode{nx, ny, new_g_cost, 0.0, current_node};
                neighbor_node->h_cost = std::hypot(nx - goal_x, ny - goal_y);
                a_star_nodes_.push_back(neighbor_node);
                all_nodes[neighbor_idx] = neighbor_node;
                open_list.push(neighbor_node);
            } else {
                // Node already exists, check if this path is better
                neighbor_node = it->second;
                if (new_g_cost < neighbor_node->g_cost) {
                    neighbor_node->g_cost = new_g_cost;
                    neighbor_node->parent = current_node;
                    open_list.push(neighbor_node);
                }
            }
        }
    }

    // --- PATH RECONSTRUCTION ---
    if (goal_node) {
        RCLCPP_INFO(this->get_logger(), "Path found after %d iterations! Reconstructing...", iterations);
        AStarNode* current = goal_node;
        while (current != nullptr) 
        {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = "map";
            pose.header.stamp = this->get_clock()->now();
            pose.pose.position.x = current->x * resolution + origin_x + resolution / 2.0; // Center in cell
            pose.pose.position.y = current->y * resolution + origin_y + resolution / 2.0; // Center in cell
            pose.pose.position.z = 0.0;
            pose.pose.orientation.w = 1.0;
            planned_path_.poses.push_back(pose);
            current = current->parent;
        }
        std::reverse(planned_path_.poses.begin(), planned_path_.poses.end());
        planned_path_.header.frame_id = "map";
        planned_path_.header.stamp = this->get_clock()->now();
        planned_path_pub_->publish(planned_path_);
        path_planned_ = true;
        RCLCPP_INFO(this->get_logger(), "Path contains %zu waypoints", planned_path_.poses.size());
    } else {
        RCLCPP_ERROR(this->get_logger(), "No path found to the goal after %d iterations!", iterations);
    }
}

void MazeSolverNode::smoothPath() {
    if (planned_path_.poses.size() < 3) {
        smoothed_path_ = planned_path_;
        RCLCPP_INFO(this->get_logger(), "Path is too short to smooth. Using original path.");
        if (!smoothed_path_.poses.empty()) {
           smoothed_path_pub_->publish(smoothed_path_);
        }
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Smoothing path...");

    smoothed_path_.header = planned_path_.header;
    smoothed_path_.poses.clear();

    // Always include the first point
    smoothed_path_.poses.push_back(planned_path_.poses[0]);

    // Reduce waypoint density by keeping only significant direction changes
    for (size_t i = 1; i < planned_path_.poses.size() - 1; ++i) {
        const auto& prev = smoothed_path_.poses.back().pose.position; // Compare to the last added point
        const auto& curr = planned_path_.poses[i].pose.position;
        const auto& next = planned_path_.poses[i+1].pose.position;

        // Calculate vectors
        double dx1 = curr.x - prev.x;
        double dy1 = curr.y - prev.y;
        double dx2 = next.x - curr.x;
        double dy2 = next.y - curr.y;

        // Calculate angle between vectors
        double dot = dx1 * dx2 + dy1 * dy2;
        double mag1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
        double mag2 = std::sqrt(dx2 * dx2 + dy2 * dy2);

        if (mag1 > 1e-6 && mag2 > 1e-6) {
            double cos_angle = dot / (mag1 * mag2);
            cos_angle = std::max(-1.0, std::min(1.0, cos_angle)); // Manual clamp
            double angle = std::acos(cos_angle);

            // Keep point if there's a significant direction change or sufficient distance
            if (angle > 0.2 || calculateDistance(prev, curr) > 0.5) {
                smoothed_path_.poses.push_back(planned_path_.poses[i]);
            }
        }
    }

    // Always include the last point
    smoothed_path_.poses.push_back(planned_path_.poses.back());

    smoothed_path_pub_->publish(smoothed_path_);
    RCLCPP_INFO(this->get_logger(), "Path smoothed from %zu to %zu waypoints",
               planned_path_.poses.size(), smoothed_path_.poses.size());
}


// void MazeSolverNode::smoothPath() {
//     if (planned_path_.poses.size() < 3) {
//         smoothed_path_ = planned_path_;
//         RCLCPP_INFO(this->get_logger(), "Path is too short to smooth. Using original path.");
//         if (!smoothed_path_.poses.empty()) {
//            smoothed_path_.header.stamp = this->get_clock()->now();
//            smoothed_path_pub_->publish(smoothed_path_);
//            rclcpp::sleep_for(std::chrono::milliseconds(200));
//         }
//         return;
//     }

//     RCLCPP_INFO(this->get_logger(), "Smoothing path...");

//     smoothed_path_.header.frame_id = planned_path_.header.frame_id;
//     smoothed_path_.header.stamp = this->get_clock()->now();
//     smoothed_path_.poses.clear();

//     // Always include the first point
//     smoothed_path_.poses.push_back(planned_path_.poses[0]);

//     // Reduce waypoint density by keeping only significant direction changes
//     for (size_t i = 1; i < planned_path_.poses.size() - 1; ++i) {
//         const auto& prev = smoothed_path_.poses.back().pose.position;
//         const auto& curr = planned_path_.poses[i].pose.position;
//         const auto& next = planned_path_.poses[i+1].pose.position;

//         // Calculate vectors
//         double dx1 = curr.x - prev.x;
//         double dy1 = curr.y - prev.y;
//         double dx2 = next.x - curr.x;
//         double dy2 = next.y - curr.y;

//         // Calculate angle between vectors
//         double dot = dx1 * dx2 + dy1 * dy2;
//         double mag1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
//         double mag2 = std::sqrt(dx2 * dx2 + dy2 * dy2);

//         if (mag1 > 1e-6 && mag2 > 1e-6) {
//             double cos_angle = dot / (mag1 * mag2);
//             cos_angle = std::max(-1.0, std::min(1.0, cos_angle)); // Manual clamp
//             double angle = std::acos(cos_angle);

//             // Keep point if there's a significant direction change or sufficient distance
//             if (angle > 0.2 || calculateDistance(prev, curr) > 0.5) {
//                 smoothed_path_.poses.push_back(planned_path_.poses[i]);
//             }
//         }
//     }

//     // Always include the last point
//     smoothed_path_.poses.push_back(planned_path_.poses.back());

//     smoothed_path_pub_->publish(smoothed_path_);
//     RCLCPP_INFO(this->get_logger(), "Path smoothed from %zu to %zu waypoints",
//                planned_path_.poses.size(), smoothed_path_.poses.size());

//     // Give the middleware a moment to publish and latch the message.
//     rclcpp::sleep_for(std::chrono::milliseconds(200));
// }


double MazeSolverNode::calculateDistance(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

void MazeSolverNode::controlLoop() {
    auto twist_msg = std::make_unique<geometry_msgs::msg::TwistStamped>();
    twist_msg->header.stamp = this->get_clock()->now();
    twist_msg->header.frame_id = "";  // Use empty frame for cmd_vel

    const auto& path_to_follow = smoothed_path_.poses.empty() ? planned_path_ : smoothed_path_;

    if (!path_planned_ || path_to_follow.poses.empty() || current_waypoint_index_ >= path_to_follow.poses.size()) {
        cmd_vel_pub_->publish(std::move(twist_msg)); // Publish zero velocity
        return;
    }

    size_t target_index = current_waypoint_index_;
    for (size_t i = current_waypoint_index_; i < path_to_follow.poses.size(); ++i) {
        double distance = calculateDistance(current_pose_.position, path_to_follow.poses[i].pose.position);
        if (distance >= lookahead_distance_) {
            target_index = i;
            break;
        }
        target_index = i;
    }

    const auto& target_pose = path_to_follow.poses[target_index].pose;
    double dx = target_pose.position.x - current_pose_.position.x;
    double dy = target_pose.position.y - current_pose_.position.y;
    double distance_to_target = std::hypot(dx, dy);
    double angle_to_target = std::atan2(dy, dx);
    double current_yaw = getYawFromQuaternion(current_pose_.orientation);
    double heading_error = angle_to_target - current_yaw;

    while (heading_error > M_PI) heading_error -= 2.0 * M_PI;
    while (heading_error < -M_PI) heading_error += 2.0 * M_PI;

    double distance_to_current_waypoint = calculateDistance(current_pose_.position,
                                                           path_to_follow.poses[current_waypoint_index_].pose.position);

    if (distance_to_current_waypoint < goal_radius_ && current_waypoint_index_ < path_to_follow.poses.size() - 1) {
        current_waypoint_index_++;
        RCLCPP_INFO(this->get_logger(), "Advanced to waypoint %zu/%zu",
                       current_waypoint_index_ + 1, path_to_follow.poses.size());
    }

    double distance_to_final_goal = calculateDistance(current_pose_.position, path_to_follow.poses.back().pose.position);
    if (distance_to_final_goal < goal_radius_) {
         RCLCPP_INFO(this->get_logger(), "Goal reached! Stopping robot.");
         path_planned_ = false;
         cmd_vel_pub_->publish(std::move(twist_msg));
         return;
    }

    // --- Smoothening the turns to avoid stopping ---

    // Angular velocity 
    double angular_vel = kp_angular_ * heading_error;
    angular_vel = std::max(-max_angular_vel_, std::min(max_angular_vel_, angular_vel));

    // Calculate a base linear velocity
    double linear_vel = kp_linear_ * distance_to_target;
    linear_vel = std::min(max_linear_vel_, linear_vel); // Don't exceed max speed

    // Scale linear velocity based on how far we need to turn.
    // The larger the heading error, the slower the robot moves forward.
    double heading_abs = std::abs(heading_error);
    if (heading_abs > 0.2) { // Only start scaling if the error is somewhat significant
        // Create a scaling factor from 1.0 (straight ahead) to 0.0 (90 degrees off)
        // M_PI_2 is pi/2, or 90 degrees.
        double scale = std::max(0.0, 1.0 - (heading_abs / M_PI_2));
        linear_vel *= scale;
    }
    
    // Always making sure that the linear velocity is positive
    linear_vel = std::max(0.0, linear_vel);
    
    twist_msg->twist.linear.x = linear_vel;
    twist_msg->twist.angular.z = angular_vel;

    cmd_vel_pub_->publish(std::move(twist_msg));
}

double MazeSolverNode::getYawFromQuaternion(const geometry_msgs::msg::Quaternion& q) {
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MazeSolverNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}