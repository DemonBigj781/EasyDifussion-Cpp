#ifndef __TASK_STATE_H__
#define __TASK_STATE_H__

#include <map>
#include <mutex>
#include <string>
#include <vector>

struct TaskState {
    std::string task_id;
    bool completed;
    float progress;
    int current_step;
    int total_steps;
    std::string live_preview;  // base64 encoded image
    int id_live_preview;
    std::vector<std::string> result_images;  // base64 encoded images
    std::string info;                        // JSON string with additional info
    bool interrupted;
};

class TaskStateManager {
   public:
    TaskStateManager();
    ~TaskStateManager();

    // Create or update a task
    void createTask(const std::string& task_id);
    void updateTaskProgress(const std::string& task_id, float progress, const std::string& live_preview = "",
                            int current_step = -1, int total_steps = -1);
    void completeTask(const std::string& task_id, const std::vector<std::string>& images, const std::string& info);
    void interruptTask(const std::string& task_id);

    // Get task state
    TaskState getTaskState(const std::string& task_id);
    bool taskExists(const std::string& task_id);

    // Clear old tasks
    void clearTask(const std::string& task_id);

   private:
    std::map<std::string, TaskState> tasks_;
    std::mutex mutex_;
};

#endif  // __TASK_STATE_H__
