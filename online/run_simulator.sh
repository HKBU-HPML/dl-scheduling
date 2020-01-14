#python job_master.py blf-1-sf-0-false > blf-1-sf-0-false.log
#python job_master.py blf-2-sf-0-false > blf-2-sf-0-false.log
#python job_master.py blf-4-sf-0-false > blf-4-sf-0-false.log
#python job_master.py blf-8-sf-0-false > blf-8-sf-0-false.log
#python job_master.py blf-16-sf-0-false > blf-16-sf-0-false.log

## effects of pick-gpu str for sf
#python job_master.py sf-bp-sf-1-true >   logs/sf-bp-sf-1-true.log
#python job_master.py sf-lgwf-sf-1-true > logs/sf-lgwf-sf-1-true.log
#python job_master.py sf-lswf-sf-1-true > logs/sf-lswf-sf-1-true.log
#
## effects of pick-gpu str for stf
#python job_master.py stf-bp-srtf-1-true >   logs/stf-bp-srtf-1-true.log
#python job_master.py stf-lgwf-srtf-1-true > logs/stf-lgwf-srtf-1-true.log
#python job_master.py stf-lswf-srtf-1-true > logs/stf-lswf-srtf-1-true.log

# effects of pick-gpu str for ssf
python job_master.py ssf-random-srsf-1-true >    logs/ssf-random-srsf-1-true.log
#python job_master.py ssf-bp-srsf-1-true >    logs/ssf-bp-srsf-1-true.log
#python job_master.py ssf-lwf1-srsf-1-true >  logs/ssf-lwf1-srsf-1-true.log
#python job_master.py ssf-lwf2-srsf-1-true >  logs/ssf-lwf2-srsf-1-true.log
#python job_master.py ssf-lwf4-srsf-1-true >  logs/ssf-lwf4-srsf-1-true.log
#python job_master.py ssf-lwf8-srsf-1-true >  logs/ssf-lwf8-srsf-1-true.log
#python job_master.py ssf-lwf32-srsf-1-true > logs/ssf-lwf32-srsf-1-true.log

## effects of adaDual
#python job_master.py ssf-2-srsf-0-false > logs/ssf-2-srsf-0-false.log
#python job_master.py ssf-2-srsf-1-false > logs/ssf-2-srsf-1-false.log
#python job_master.py ssf-2-srsf-2-false > logs/ssf-2-srsf-2-false.log
#python job_master.py ssf-2-srsf-1-true  > logs/ssf-2-srsf-1-true.log
