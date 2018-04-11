import subprocess
import click

@click.command()
@click.option('--asset', '-a', multiple=True)
@click.option('--msg/--no-msg', default = False, help = 'Enables/disables telegram messaging. Defalut False')
@click.option('--dev', is_flag = True, help = 'Development flag')
def main(asset, msg, dev):
    norms = ['mapminmax', 'mapstd']
    for a in asset:
        for norm in norms:
            args = ['python', 'TrainMLP.py', '--asset={}'.format(a), '--inits=10', '--norm={}'.format(norm), '--optimizer=adam', '--verbose']
            args.append('--msg') if msg else args.append('--no-msg')
            if dev: args.append('--dev')
            #print args
            p = subprocess.Popen(args)
            p.wait()

if __name__ == '__main__':
    main()
