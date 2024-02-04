import numpy as np
import pandas as pd
import paramiko as pm
import socket as socket
import os

class Job:
    '''
    Job class defines name of folder and loads POSCAR file. Methods are provided for loading and modifying locally saved reference INCAR and KPOINTS files.

    Parameters
    ----------
    name:    str,
        Name of job which also is going to be the remote folder name.
    poscar_path:   str
        Path to local POSCAR file.
    '''
    def __init__(self, name: str, poscar_path_local: str):
        self.name = name
        self.poscar_path = poscar_path_local
        self.elements = np.loadtxt(poscar_path_local,dtype=str,skiprows=5, max_rows=1)
        print(f'{self.name} Job instance started')

    def incar_load(self, incar_path_reference: str):
        '''Load reference INCAR file as pandas data frame'''
        self.incar = pd.read_csv(incar_path_reference, names=['Setting','Value'], sep=' = ', dtype=str, engine='python')
    
    def incar_modify(self, setting: str, value: str):
        '''Modify individual values of a setting after incar is loaded into instance'''
        self.incar.loc[self.incar['Setting'] == setting,'Value'] = value

    def kpoints_load(self, kpoints_path_reference: str):
        '''Load reference KPOINTS file as pandas data frame'''
        self.kpoints = pd.read_csv(kpoints_path_reference, names=['Value','Description'], sep=' ! ', dtype=str, engine='python')
    
    def kpoints_modify(self, description: str, value: str):
        '''Modify individual values after kpoints is loaded into instance'''
        self.kpoints.loc[self.kpoints['Description'] == description,'Value'] = value

class Server:
    '''Server class offers methods for connecting to the server, creating folders, and uploading files'''
    def __init__(self, username: str, host: str, home_path_remote: str, batch_path_remote: str):
        self.host = host
        self.user = username
        self.home_path = home_path_remote # SFTP home path
        self.batch_path = batch_path_remote # e.g. ~/batch_scripts/vasp.sbatch

    def connect(self):
        '''Connects to server and returns SFTP and transport objects'''
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, 22))
        ts = pm.Transport(sock)
        ts.start_client(timeout=10)
        ts.auth_interactive_dumb(self.user)
        print(ts)
        sftp = pm.SFTPClient.from_transport(ts)
        print(sftp)
        return sftp, ts
        
    def create_folder(self, sftp: object, name: str):
        '''Construct folder in home directory'''
        sftp.mkdir(name)

    def potcar_gather(self, job: Job, sftp:object, potcar_folder_path: str, suffix_to_element:str=''):
        '''
        Gathers and appends POTCAR files for the elements. Handling is server-side.

        Parameters
        ----------
        job:    Job
            Job instance
        sftp: object
            Paramiko SFTP object
        potcar_folder_path:   str
            Remote path to POTCAR folder. Requires potential files contained there to be named after their respective Element (e.g. Li).
        suffix_to_element:  str=''
            Added suffix to name of potentials. E.g. to choose X_GW potentials, assign "_GW".
        '''
        file = sftp.open(f'{potcar_folder_path}/{job.elements[0]}/POTCAR', 'r') # open POTCAR for first element in reading mode
        sftp.putfo(file, f'{self.home_path}/{job.name}/POTCAR', confirm=True) # save as POTCAR in job folder
        print(f'Created POTCAR for {job.elements[0]}')
        file.close() # close initial file
        file = sftp.file(f'{self.home_path}/{job.name}/POTCAR', 'a') # open new POTCAR file in appending mode

        for element in job.elements[1:]:
            path = f'{potcar_folder_path}/{element}{suffix_to_element}/POTCAR'
            sftp.getfo(path, file)
            print(f'Appended POTCAR for {element}')
            
        file.close() # close final INCAR file
        print('POTCAR files successfully merged and copied to job folder')
        
    def incar_upload(self, job: Job, sftp:object, tempfolder_path_local: str):
        '''Upload local INCAR file, must be loaded by Job.incar_load first. Requires assignment of a folder where files are temporarily saved before upload.'''
        np.savetxt(f'{tempfolder_path_local}/INCAR',job.incar, fmt='%1s', delimiter=' = ')
        print('INCAR file temporarily saved')
        
        try:
            sftp.put(f'{tempfolder_path_local}/INCAR', f'{self.home_path}/{job.name}/INCAR', confirm=True)
            print('INCAR file uploaded')
        except Exception as e:
            print(f'Failed to upload INCAR file. {e}')

        os.remove(f'{tempfolder_path_local}/INCAR')
        if not os.path.exists(f'{tempfolder_path_local}/INCAR'): print('Temporarirly saved INCAR removed')

    def poscar_upload(self, job: Job, sftp: object):
        '''Upload local POSCAR file'''
        try:
            sftp.put(job.poscar_path, f'{self.home_path}/{job.name}/POSCAR', confirm=True)
            print('POSCAR file uploaded')
        except Exception as e:
            print(f'Failed to upload POSCAR file. {e}')
        
    def kpoints_upload(self, job: Job, sftp: object, tempfolder_path_local: str):
        '''Upload local KPOINTS file, must be loaded by Job.kpoints_load first'''
        np.savetxt(f'{tempfolder_path_local}/KPOINTS', job.kpoints['Value'], fmt='%1s', delimiter=' ! ')
        print('KPOINTS file temporarily saved')

        try:
            sftp.put(tempfolder_path_local+'/KPOINTS', f'{self.home_path}/{job.name}/KPOINTS', confirm=True)
            print('KPOINTS file uploaded')
        except Exception as e:
            print(f'Failed to upload KPOINTS file. {e}')

        os.remove(f'{tempfolder_path_local}/KPOINTS')
        if not os.path.exists(tempfolder_path_local+'/KPOINTS'): print('Temporarirly saved KPOINTS removed')

    def name_batch(self, job:Job, ts: object):
        '''Give name for job to .sbatch file - needs #SBATCH -J line which specifies name beforehand'''
        ch= ts.open_session(timeout=15)
        ch.exec_command(f"sed -i '/SBATCH -J/c\#SBATCH -J {job.name}' {self.batch_path}")
        ret = ch.recv_exit_status()
        ch.close()
        print(f'Updating name - Exit status of execution command: {ret}')

    def batch_submit(self, job: Job, ts: object):
        '''Submit batch for calculation'''
        ch= ts.open_session(timeout=15)
        ch.exec_command(f'cd ~/{job.name} \n sbatch {self.batch_path}')
        ret = ch.recv_exit_status()
        ch.close()
        print(f'Submitting batch - Exit status of execution command: {ret}')

    def exec_command(self, ts: object, command: str):
        '''Execution of a single SSH command while obtaining list of returns.
        
        Returns
        ----------
        answers: list
            Received returns from server.'''
        key = True
        answers=[]
        chan= ts.open_session(timeout=15)

        chan.exec_command(command)
        while key:
            if chan.recv_ready():
                answer = chan.recv(4096).decode('ascii')
                answers.append(answer)
                print("recv:\n%s" % answer)
            if chan.recv_stderr_ready():
                answer = chan.recv_stderr(4096).decode('ascii')
                answers.append(answer)
                print("error:\n%s" % answer)
            if chan.exit_status_ready():
                print("exit status: %s" % chan.recv_exit_status())
                key = False

        return answers