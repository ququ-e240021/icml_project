# ------------------------------------ REQUIRED LINE # 1 ------------------------------------
# now, create the RigLScheduler object
pruner = RigLScheduler(model,                           # model you created
                       optimizer,                       # optimizer (recommended = SGD w/ momentum)
                       dense_allocation=0.1,            # a float between 0 and 1 that designates how sparse you want the network to be 
                                                          # (0.1 dense_allocation = 90% sparse)
                       sparsity_distribution='uniform', # distribution hyperparam within the paper, currently only supports `uniform`
                       T_end=T_end,                     # T_end hyperparam within the paper (recommended = 75% * total_iterations)
                       delta=100,                       # delta hyperparam within the paper (recommended = 100)
                       alpha=0.3,                       # alpha hyperparam within the paper (recommended = 0.3)
                       grad_accumulation_n=1,           # new hyperparam contribution (not in the paper) 
                                                          # for more information, see the `Contributions Beyond the Paper` section
                       static_topo=False,               # if True, the topology will be frozen, in other words RigL will not do it's job 
                                                          # (for debugging)
                       ignore_linear_layers=False,      # if True, linear layers in the network will be kept fully dense
                       state_dict=None)                 # if you have checkpointing enabled for your training script, you should save 
                                                          # `pruner.state_dict()` and when resuming pass the loaded `state_dict` into 
                                                          # the pruner constructor
# -------------------------------------------------------------------------------------------
                       
... more code ...

for epoch in range(epochs):
    for data in dataloader:
        # do forward pass, calculate loss, etc.
        ...
    
        # instead of calling optimizer.step(), wrap it as such:
    
# ------------------------------------ REQUIRED LINE # 2 ------------------------------------
        if pruner():
# -------------------------------------------------------------------------------------------
            # this block of code will execute according to the given hyperparameter schedule
            # in other words, optimizer.step() is not called after a RigL step
            optimizer.step()
        
    # it is also recommended that after every epoch you checkpoint your training progress
    # to do so with RigL training you should also save the pruner object state_dict
    torch.save({
        'model': model.state_dict(),
        'pruner': pruner.state_dict(),
        'optimizer': optimizer.state_dict()
    }, 'checkpoint.pth')
        
# at any time you can print the RigLScheduler object and it will show you the sparsity distributions, number of training steps/rigl steps, etc!
print(pruner)

# save model
torch.save(model.state_dict(), 'model.pth')
```
