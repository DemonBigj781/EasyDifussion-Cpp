#include "task_state.h"

TaskStateManager::TaskStateManager() {}

TaskStateManager::~TaskStateManager() {}

void TaskStateManager::createTask(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    TaskState state;
    state.task_id = task_id;
    state.completed = false;
    state.progress = 0.0f;
    state.current_step = 0;
    state.total_steps = 0;
    state.live_preview = "";
    state.id_live_preview = 0;
    state.interrupted = false;

    tasks_[task_id] = state;
}

void TaskStateManager::updateTaskProgress(const std::string& task_id, float progress, const std::string& live_preview,
                                          int current_step, int total_steps) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
        it->second.progress = progress;
        if (current_step >= 0) {
            it->second.current_step = current_step;
        }
        if (total_steps > 0) {
            it->second.total_steps = total_steps;
        }
        if (!live_preview.empty()) {
            it->second.live_preview = live_preview;
            it->second.id_live_preview++;
        }
    }
}

void TaskStateManager::completeTask(const std::string& task_id, const std::vector<std::string>& images,
                                    const std::string& info) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
        it->second.completed = true;
        it->second.progress = 1.0f;
        if (it->second.total_steps > 0) {
            it->second.current_step = it->second.total_steps;
        }
        it->second.result_images = images;
        it->second.info = info;
    }
}

void TaskStateManager::interruptTask(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
        it->second.interrupted = true;
    }
}

TaskState TaskStateManager::getTaskState(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
        return it->second;
    }

    // Return empty task state if not found
    TaskState empty_state;
    empty_state.task_id = task_id;
    empty_state.completed = false;
    empty_state.progress = 0.0f;
    empty_state.current_step = 0;
    empty_state.total_steps = 0;
    empty_state.id_live_preview = 0;
    empty_state.interrupted = false;

    return empty_state;
}

TaskState TaskStateManager::getTaskProgressState(const std::string& task_id,
                                                 bool include_live_preview,
                                                 int known_live_preview_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    TaskState progress_state{};
    progress_state.task_id = task_id;
    auto it = tasks_.find(task_id);
    if (it == tasks_.end()) {
        return progress_state;
    }

    const TaskState& stored = it->second;
    progress_state.completed = stored.completed;
    progress_state.progress = stored.progress;
    progress_state.current_step = stored.current_step;
    progress_state.total_steps = stored.total_steps;
    progress_state.id_live_preview = stored.id_live_preview;
    progress_state.interrupted = stored.interrupted;
    if (include_live_preview && known_live_preview_id != stored.id_live_preview) {
        progress_state.live_preview = stored.live_preview;
    }
    return progress_state;
}

bool TaskStateManager::taskExists(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.find(task_id) != tasks_.end();
}

void TaskStateManager::clearTask(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.erase(task_id);
}
