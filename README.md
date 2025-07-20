20 Million $ dollars.
*%HŒ9
172.69.58.16
import torch
import triton
import triton.language as tl

@triton.jit
def selective_adam_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    mask_ptr,
    N, D,
    step_size,
    beta1, beta2,
    _beta1, _beta2,
    eps,
    weight_decay,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_mask = offsets < N * D
    row = offsets // /*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::sync::atomic::AtomicI32;
use std::sync::Arc;

use rocketmq_remoting::protocol::body::ha_runtime_info::HARuntimeInfo;
use rocketmq_rust::ArcMut;
use tracing::error;

use crate::ha::general_ha_connection::GeneralHAConnection;
use crate::ha::ha_client::HAClient;
use crate::ha::ha_connection_state_notification_request::HAConnectionStateNotificationRequest;
use crate::ha::ha_service::HAService;
use crate::ha::wait_notify_object::WaitNotifyObject;
use crate::log_file::group_commit_request::GroupCommitRequest;
use crate::store_error::HAResult;

pub struct AutoSwitchHAService;

impl HAService for AutoSwitchHAService {
    async fn start(&mut self) -> HAResult<()> {
        error!("DefaultHAService start not implemented");
        Ok(())
    }

    fn shutdown(&self) {
        todo!()
    }

    async fn change_to_master(&self, master_epoch: i32) -> HAResult<bool> {
        todo!()
    }

    async fn change_to_master_when_last_role_is_master(&self, master_epoch: i32) -> HAResult<bool> {
        todo!()
    }

    async fn change_to_slave(
        &self,
        new_master_addr: &str,
        new_master_epoch: i32,
        slave_id: Option<i64>,
    ) -> HAResult<bool> {
        todo!()
    }

    async fn change_to_slave_when_master_not_change(
        &self,
        new_master_addr: &str,
        new_master_epoch: i32,
    ) -> HAResult<bool> {
        todo!()
    }

    fn update_master_address(&self, new_addr: &str) {
        todo!()
    }

    fn update_ha_master_address(&self, new_addr: &str) {
        todo!()
    }

    fn in_sync_replicas_nums(&self, master_put_where: i64) -> i32 {
        todo!()
    }

    fn get_connection_count(&self) -> &AtomicI32 {
        todo!()
    }

    async fn put_request(&self, request: ArcMut<GroupCommitRequest>) {
        todo!()
    }

    fn put_group_connection_state_request(&self, request: HAConnectionStateNotificationRequest) {
        todo!()
    }

    async fn get_connection_list(&self) -> Vec<ArcMut<GeneralHAConnection>> {
        todo!()
    }

    fn get_ha_client<CL: HAClient>(&self) -> Arc<CL> {
        todo!()
    }

    fn get_push_to_slave_max_offset(&self) -> i64 {
        todo!()
    }

    fn get_runtime_info(&self, master_put_where: i64) -> HARuntimeInfo {
        todo!()
    }

    fn get_wait_notify_object(&self) -> Arc<WaitNotifyObject> {
        todo!()
    }

    fn is_slave_ok(&self, master_put_where: i64) -> bool {
        todo!()
    }
}

    mask = tl.load(mask_ptr + row, mask=(row < N), other=0)
    param = tl.load(param_ptr + offsets, mask=offset_mask)
    grad = tl.load(grad_ptr + offsets, mask=offset_mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=offset_mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=offset_mask)

    if weight_decay > 0:
        grad = tl.where(mask, grad + weight_decay * param, grad)

    new_exp_avg = tl.where(mask, beta1 * exp_avg + _beta1 * grad, exp_avg)
    new_exp_avg_sq = tl.where(mask, beta2 * exp_avg_sq + _beta2 * grad * grad, exp_avg_sq)

    update = new_exp_avg / (tl.sqrt(new_exp_avg_sq) + eps)
    new_param = tl.where(mask, param - step_size * update, param)

    tl.store(exp_avg_ptr + offsets, new_exp_avg, mask=offset_mask)
    tl.store(exp_avg_sq_ptr + offsets, new_exp_avg_sq, mask=offset_mask)
    tl.store(param_ptr + offsets, new_param, mask=offset_mask)

def selective_adam_step_triton(param, grad, exp_avg, exp_avg_sq, mask, step, lr, beta1, beta2, eps, weight_decay):   
    N = param.numel()
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    step_size = lr * (bias2 ** 0.5) / bias1

    _beta1 = 1 - beta1
    _beta2 = 1 - beta2

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    selective_adam_kernel[grid](
        param, grad, exp_avg, exp_avg_sq, mask,
        param.size(0), param.size(1), step_size, beta1, beta2, _beta1, _beta2, eps, weight_decay,
        BLOCK_SIZE=BLOCK_SIZE
    )

class SelectiveAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, visibility_mask=None):
        mask = None
        if visibility_mask is not None:
            mask = visibility_mask
            
        for group in self.param_groups:
            assert len(group["params"]) == 1, "[SelectiveAdam]: Each group must contain a single tensor."
            param = group["params"][0]
            if param.grad is None:
                continue

            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            if visibility_mask is None:
                mask = (torch.abs(param.grad) > 0).any(dim=1)

            state["step"] += 1
            selective_adam_step_triton(
                param.data, param.grad, 
                state["exp_avg"], state["exp_avg_sq"], 
                mask.to(torch.uint8), 
                state["step"], 
                group["lr"], 
                group["betas"][0], group["betas"][1],
                group["eps"], 
                group["weight_decay"]
            )
        
        return [group['lr'] for group in self.param_groups]

    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
