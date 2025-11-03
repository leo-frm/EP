import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simpson_rule
from scipy.integrate import trapezoid

# =============================================================================
# FUNÇÕES DO PROBLEMA 
# =============================================================================

def p(x):
    """Coeficiente p(x) - assumimos constante = 1"""
    return 1.0

def r(x, Nu):
    """Coeficiente r(x) - número de Nusselt (constante)"""
    return Nu

def f(x, f0):
    """Termo fonte f(x) - constante"""
    return f0

def b(x):
    """
    Coeficiente b(x) relacionado à radiação
    Para simplificar, assumimos b(x) = 1 (constante)
    """
    return 1.0

def g(x, y):
    """
    Função g(x,y) - termo de radiação não-linear
    g(x, y) = b(x) * (1 + y)^4
    """
    return b(x) * (1 + y)**4

def dg_dy(x, y):
    """
    Derivada parcial de g em relação a y
    dg/dy = b(x) * 4 * (1 + y)^3
    """
    return b(x) * 4 * (1 + y)**3

# =============================================================================
# MÉTODOS NUMÉRICOS (1)
# =============================================================================

def euler_explicito(x_vec, y0, dy0, Nu=1.0, f0=0.0):
    """
    Método de Euler Explícito
    """
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    # Criar vetores para y e z
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    
    # Condições iniciais
    y[0] = y0
    z[0] = dy0
    
    # Loop de integração
    for k in range(n):
        x_k = x_vec[k]
        
        # Atualizar y usando: y' = z
        y[k+1] = y[k] + h * z[k]
        
        # Calcular dz/dx usando: z' = (r*y - f) / p
        dz_dx = (r(x_k, Nu) * y[k] - f(x_k, f0)) / p(x_k)
        
        # Atualizar z
        z[k+1] = z[k] + h * dz_dx
    
    return y

def euler_implicito(x_vec, y0, dy0, Nu=1.0, f0=0.0):
    """
    Método de Euler Implícito
    """
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    # Criar vetores para y e z
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    
    # Condições iniciais
    y[0] = y0
    z[0] = dy0
    
    # Loop de integração
    for k in range(n):
        x_k1 = x_vec[k + 1]
        
        # Calcular dz/dx no ponto k+1 usando: z' = (r*y - f) / p
        # Rearranjar para encontrar y[k+1] e z[k+1]
        # Sistema de equações:
        # y[k+1] = y[k] + h * z[k+1]
        # z[k+1] = z[k] + h * ((r(x_k1, Nu) * y[k+1] - f(x_k1, f0)) / p(x_k1))
        
        # Resolver o sistema linear
        A = np.array([[1, -h],
                      [-h * r(x_k1, Nu) / p(x_k1), 1]])
        
        b = np.array([y[k],
                      z[k] + h * f(x_k1, f0) / p(x_k1)])
        
        sol = np.linalg.solve(A, b)
        
        y[k + 1] = sol[0]
        z[k + 1] = sol[1]
    
    return y


def trapezio_explicito(x_vec, y0, dy0, Nu=1.0, f0=0.0):
    """
    Método do Trapézio Explícito
    """
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    # Criar vetores para y e z
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    
    # Condições iniciais
    y[0] = y0
    z[0] = dy0
    
    # Loop de integração
    for k in range(n):
        x_k = x_vec[k]
        x_k1 = x_vec[k + 1]
        
        # Calcular dz/dx no ponto k usando: z' = (r*y - f) / p
        dz_dx_k = (r(x_k, Nu) * y[k] - f(x_k, f0)) / p(x_k)
        
        # Prever y e z no ponto k+1 usando Euler explícito
        y_pred = y[k] + h * z[k]
        z_pred = z[k] + h * dz_dx_k
        
        # Calcular dz/dx no ponto k+1 usando: z' = (r*y - f) / p
        dz_dx_k1 = (r(x_k1, Nu) * y_pred - f(x_k1, f0)) / p(x_k1)
        
        # Atualizar y e z usando a média dos dois pontos
        y[k + 1] = y[k] + (h / 2) * (z[k] + z_pred)
        z[k + 1] = z[k] + (h / 2) * (dz_dx_k + dz_dx_k1)
    
    return y

def trapezio_implicito(x_vec, y0, dy0, Nu=1.0, f0=0.0):
    """
    Método do Trapézio Implícito
    """
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    # Criar vetores para y e z
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    
    # Condições iniciais
    y[0] = y0
    z[0] = dy0
    
    # Loop de integração
    for k in range(n):
        x_k1 = x_vec[k + 1]
        
        # Calcular dz_dx no ponto k (explícito)
        dz_k = (r(x_vec[k], Nu) * y[k] - f(x_vec[k], f0)) / p(x_vec[k])

        # Sistema linear
        A = np.array([[1, -h/2],
                    [-h/2 * r(x_k1, Nu) / p(x_k1), 1]])

        b = np.array([y[k] + (h/2) * z[k],
                    z[k] + (h/2) * (dz_k + f(x_k1, f0) / p(x_k1))])

        sol = np.linalg.solve(A, b)
        
        y[k + 1] = sol[0]
        z[k + 1] = sol[1]
    
    return y

def runge_kutta_4(x_vec, y0, dy0, Nu=1.0, f0=0.0):
    """
    Método de Runge-Kutta de 4ª ordem
    """
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    
    y[0] = y0
    z[0] = dy0
    
    for k in range(n):
        x_k = x_vec[k]
        y_k = y[k]
        z_k = z[k]
        
        # k1
        k1_y = z_k
        k1_z = (r(x_k, Nu) * y_k - f(x_k, f0)) / p(x_k)
        
        # k2 (avaliar em x_k + h/2)
        k2_y = z_k + h/2 * k1_z
        k2_z = (r(x_k + h/2, Nu) * (y_k + h/2 * k1_y) - f(x_k + h/2, f0)) / p(x_k + h/2)
        
        # k3 (avaliar em x_k + h/2)
        k3_y = z_k + h/2 * k2_z
        k3_z = (r(x_k + h/2, Nu) * (y_k + h/2 * k2_y) - f(x_k + h/2, f0)) / p(x_k + h/2)
        
        # k4 (avaliar em x_k + h)
        k4_y = z_k + h * k3_z
        k4_z = (r(x_k + h, Nu) * (y_k + h * k3_y) - f(x_k + h, f0)) / p(x_k + h)
        
        # Atualizar
        y[k+1] = y_k + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        z[k+1] = z_k + h/6 * (k1_z + 2*k2_z + 2*k3_z + k4_z)
    
    return y

# =============================================================================
# MÉTODOS NUMÉRICOS (2)
# =============================================================================

def rk4_nao_linear(x_vec, y0, dy0, Nu=1.0, f0=0.0, epsilon=1.0):
    """
    Método de Runge-Kutta de 4ª ordem para problemas não lineares
    """
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    
    y[0] = y0
    z[0] = dy0
    
    for k in range(n):
        x_k = x_vec[k]
        y_k = y[k]
        z_k = z[k]
        
        # k1
        k1_y = z_k
        k1_z = (r(x_k, Nu) * y_k + epsilon * g(x_k, y_k) - f(x_k, f0)) / p(x_k)
        
        # k2 (avaliar em x_k + h/2)
        y_temp = y_k + h/2 * k1_y
        z_temp = z_k + h/2 * k1_z
        k2_y = z_temp
        k2_z = (r(x_k + h/2, Nu) * y_temp + epsilon * g(x_k + h/2, y_temp) - f(x_k + h/2, f0)) / p(x_k + h/2)
        
        # k3 (avaliar em x_k + h/2)
        y_temp = y_k + h/2 * k2_y
        z_temp = z_k + h/2 * k2_z
        k3_y = z_temp
        k3_z = (r(x_k + h/2, Nu) * y_temp + epsilon * g(x_k + h/2, y_temp) - f(x_k + h/2, f0)) / p(x_k + h/2)
        
        # k4 (avaliar em x_k + h)
        y_temp = y_k + h * k3_y
        z_temp = z_k + h * k3_z
        k4_y = z_temp
        k4_z = (r(x_k + h, Nu) * y_temp + epsilon * g(x_k + h, y_temp) - f(x_k + h, f0)) / p(x_k + h)
        
        # Atualizar
        y[k+1] = y_k + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        z[k+1] = z_k + h/6 * (k1_z + 2*k2_z + 2*k3_z + k4_z)
    
    return y

def rk4_linearizado(x_vec, z0, dz0, y_vec, Nu=1.0, epsilon=1.0):
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    z = np.zeros(n + 1)
    w = np.zeros(n + 1)
    
    z[0] = z0
    w[0] = dz0
    
    for k in range(n):
        x_k = x_vec[k]
        z_k = z[k]
        w_k = w[k]
        y_k = y_vec[k]
        
        # k1
        k1_z = w_k
        k1_w = (r(x_k, Nu)*z_k + epsilon*dg_dy(x_k, y_k)*z_k) / p(x_k)
        
        # k2
        z_temp = z_k + h/2 * k1_z
        w_temp = w_k + h/2 * k1_w
        y_temp = y_vec[k]  # simplificação: usar y no ponto k
        
        k2_z = w_temp
        k2_w = (r(x_k + h/2, Nu)*z_temp + epsilon*dg_dy(x_k + h/2, y_temp)*z_temp) / p(x_k + h/2)
        
        # k3
        z_temp = z_k + h/2 * k2_z
        w_temp = w_k + h/2 * k2_w
        y_temp = y_vec[k]
        
        k3_z = w_temp
        k3_w = (r(x_k + h/2, Nu)*z_temp + epsilon*dg_dy(x_k + h/2, y_temp)*z_temp) / p(x_k + h/2)
        
        # k4
        z_temp = z_k + h * k3_z
        w_temp = w_k + h * k3_w
        y_temp = y_vec[min(k+1, n)]  # tentar pegar o próximo ponto se possível
        
        k4_z = w_temp
        k4_w = (r(x_k + h, Nu)*z_temp + epsilon*dg_dy(x_k + h, y_temp)*z_temp) / p(x_k + h)
        
        # Atualizar
        z[k+1] = z_k + h/6 * (k1_z + 2*k2_z + 2*k3_z + k4_z)
        w[k+1] = w_k + h/6 * (k1_w + 2*k2_w + 2*k3_w + k4_w)
    
    return z

# =============================================================================
# TESTE (exemplo de uso)
# =============================================================================

'''
if __name__ == "__main__":
    print("="*70)
    print("TESTANDO OS MÉTODOS NUMÉRICOS")
    print("="*70)
    
    # Parâmetros do teste
    n = 100
    x_vec = np.linspace(0, 1, n + 1)
    
    # Condições iniciais
    y0 = 0.1
    dy0 = 0.1  # chute inicial para y'(0)
    
    # Parâmetros do problema
    Nu = 1.0
    f0 = 0.0
    epsilon = 0.0  # começar com caso linear
    
    print(f"\nParâmetros:")
    print(f"  n = {n} pontos")
    print(f"  y(0) = {y0}")
    print(f"  y'(0) = {dy0}")
    print(f"  Nu = {Nu}")
    print(f"  f0 = {f0}")
    
    # Testar todos os métodos LINEARES
    print("\n" + "-"*70)
    print("MÉTODOS LINEARES (Tarefa Computacional 1)")
    print("-"*70)
    
    y_euler_exp = euler_explicito(x_vec, y0, dy0, Nu, f0)
    print(f"Euler Explícito:     y(0.5) = {y_euler_exp[n//2]:.6f}  |  y(1) = {y_euler_exp[-1]:.6f}")
    
    y_euler_imp = euler_implicito(x_vec, y0, dy0, Nu, f0)
    print(f"Euler Implícito:     y(0.5) = {y_euler_imp[n//2]:.6f}  |  y(1) = {y_euler_imp[-1]:.6f}")
    
    y_trap_exp = trapezio_explicito(x_vec, y0, dy0, Nu, f0)
    print(f"Trapézio Explícito:  y(0.5) = {y_trap_exp[n//2]:.6f}  |  y(1) = {y_trap_exp[-1]:.6f}")
    
    y_trap_imp = trapezio_implicito(x_vec, y0, dy0, Nu, f0)
    print(f"Trapézio Implícito:  y(0.5) = {y_trap_imp[n//2]:.6f}  |  y(1) = {y_trap_imp[-1]:.6f}")
    
    y_rk4 = runge_kutta_4(x_vec, y0, dy0, Nu, f0)
    print(f"Runge-Kutta 4:       y(0.5) = {y_rk4[n//2]:.6f}  |  y(1) = {y_rk4[-1]:.6f}")
    
    # Testar método NÃO-LINEAR
    print("\n" + "-"*70)
    print("MÉTODOS NÃO-LINEARES (Tarefa Computacional 2)")
    print("-"*70)
    
    # Teste 1: epsilon = 0 (deve dar igual ao RK4 linear)
    epsilon = 0.0
    y_nl_0 = rk4_nao_linear(x_vec, y0, dy0, Nu, f0, epsilon)
    print(f"RK4 Não-Linear (ε=0): y(0.5) = {y_nl_0[n//2]:.6f}  |  y(1) = {y_nl_0[-1]:.6f}")
    print(f"  → Diferença com RK4 linear: {abs(y_nl_0[-1] - y_rk4[-1]):.2e}")
    
    # Teste 2: epsilon = 1 (problema não-linear completo)
    epsilon = 1.0
    y_nl_1 = rk4_nao_linear(x_vec, y0, dy0, Nu, f0, epsilon)
    print(f"RK4 Não-Linear (ε=1): y(0.5) = {y_nl_1[n//2]:.6f}  |  y(1) = {y_nl_1[-1]:.6f}")
    
    # Testar método LINEARIZADO
    print("\n" + "-"*70)
    print("MÉTODO LINEARIZADO")
    print("-"*70)
    
    # Condições para o linearizado
    z0 = 0.0
    dz0 = 1.0
    
    z_lin = rk4_linearizado(x_vec, z0, dz0, y_nl_1, Nu, epsilon)
    print(f"RK4 Linearizado:     z(0.5) = {z_lin[n//2]:.6f}  |  z(1) = {z_lin[-1]:.6f}")
    
    # Verificação: z deve começar em 0
    print(f"  → z(0) = {z_lin[0]:.6f} (deve ser 0.0)")
    
    print("\n" + "="*70)
    print("TESTES CONCLUÍDOS!")
    print("="*70)
    '''

# =============================================================================
# TAREFA NUMÉRICA 1
# =============================================================================

def metodoDisparo(x_vec, alpha, beta, Nu, f0, metodo):

    # definição do metodo
    if metodo == 'euler_exp':
        resolver = euler_explicito
    elif metodo == 'euler_imp':
        resolver = euler_implicito
    elif metodo == 'trap_exp':
        resolver = trapezio_explicito
    elif metodo == 'trap_imp':
        resolver = trapezio_implicito
    else:  # 'rk4'
        resolver = runge_kutta_4
    
    # Resolver problema auxiliar 1: v(0)=1, v'(0)=0
    v = resolver(x_vec, 1.0, 0.0, Nu, 0.0)  # f=0 para homogêneo
    
    # Resolver problema auxiliar 2: w(0)=0, w'(0)=1
    w = resolver(x_vec, 0.0, 1.0, Nu, 0.0)  # f=0 para homogêneo    

    # Resolver problema particular se f0 != 0
    if abs(f0) > 1e-12:
        u_p = resolver(x_vec, 0.0, 0.0, Nu, f0)
    else:
        u_p = np.zeros_like(v)
    
    # Determinar coeficientes a e b
    a = alpha - u_p[0]
    b = (beta - a * v[-1] - u_p[-1]) / w[-1]
    
    # Solução final
    y = a * v + b * w + u_p
    
    return y

def solucaoExata(x, alpha, beta, Nu):
    sqrt_nu = np.sqrt(Nu)
    
    # Calcular exponenciais
    exp_neg = np.exp(-sqrt_nu)
    exp_pos = np.exp(sqrt_nu)
    
    # Resolver para B
    B = (beta - alpha * exp_neg) / (exp_pos - exp_neg)
    
    # Resolver para A
    A = alpha - B
    
    # Calcular y_ex(x)
    y_ex = A * np.exp(-sqrt_nu * x) + B * np.exp(sqrt_nu * x)
    
    return y_ex

def calcularErro(y_numerica, y_exata):
    erroQuadratico = np.sqrt(np.mean((y_numerica - y_exata)**2))
    return erroQuadratico

def tarefaNumerica1(h, Nu, metodo, alpha=0.1, beta=0.5, f0=0.0, epsilon=0.0):
    # Parâmetros do teste
    n = int(1 / h)
    x_vec = np.linspace(0, 1, n + 1)

    # Solução exata
    y_exata = solucaoExata(x_vec, alpha, beta, Nu)
    
    # Solução numérica via método de disparo
    y_numerica = metodoDisparo(x_vec, alpha, beta, Nu, f0, metodo)

    # Cálculo do erro
    erro = calcularErro(y_numerica, y_exata)

    return y_numerica, y_exata, erro

def casosTarefaNumerica1():
    # Parâmetros para teste
    Nu_valores = [1, 16, 256]
    h_valores = [0.5, 0.05, 0.005, 0.0005]
    metodos = ['euler_exp', 'euler_imp', 'trap_exp', 'trap_imp', 'rk4']
    
    # Armazenar todos os resultados
    resultados = {}
    
    for Nu in Nu_valores:
        resultados[Nu] = {}
        for h in h_valores:
            resultados[Nu][h] = {}
            for metodo in metodos:
                y_num, y_ex, erro = tarefaNumerica1(h, Nu, metodo)
                resultados[Nu][h][metodo] = {
                    'erro': erro,
                    'y_numerica': y_num,
                    'y_exata': y_ex
                }
    
    return resultados

    import matplotlib.pyplot as plt

def plotarGraficosConvergencia(resultados):
    Nu_valores = [1, 16, 256]
    h_valores = [0.5, 0.05, 0.005, 0.0005]
    metodos = ['euler_exp', 'euler_imp', 'trap_exp', 'trap_imp', 'rk4']
    
    nomes_metodos = {
        'euler_exp': 'Euler Explícito',
        'euler_imp': 'Euler Implícito',
        'trap_exp': 'Trapézio Explícito',
        'trap_imp': 'Trapézio Implícito',
        'rk4': 'Runge-Kutta 4'
    }
    
    # Criar uma figura com 3 subplots (um para cada Nu)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, Nu in enumerate(Nu_valores):
        ax = axs[idx]
        
        # Para cada método, coletar erros e plotar
        for metodo in metodos:
            erros = []
            for h in h_valores:
                erro = resultados[Nu][h][metodo]['erro']
                erros.append(erro)
            
            # Plotar em escala log-log
            ax.loglog(h_valores, erros, 'o-', label=nomes_metodos[metodo])
        
        ax.set_xlabel('h (tamanho do passo)')
        ax.set_ylabel('Erro')
        ax.set_title(f'Nu = {Nu}')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergencia_tarefa1.png', dpi=300, bbox_inches='tight')
    plt.show()

def calcularFluxoCalor(x_vec, y, Nu):
    # Integrando
    integrando = Nu * y  # r(x) * y(x), com r(x) = Nu
    
    # Método dos trapézios
    fluxo_trapezios = trapezoid(integrando, x_vec)
    
    # Método de Simpson
    fluxo_simpson = simpson_rule(integrando, x=x_vec)
    
    return fluxo_trapezios, fluxo_simpson

def analisarFluxosCalor(resultados):
    Nu_valores = [1, 16, 256]
    h_valores = [0.5, 0.05, 0.005, 0.0005]
    
    for Nu in Nu_valores:
        # print(f"\n{'='*90}")
        # print(f"Nu = {Nu}")
        # print(f"{'='*90}")
        # print(f"{'h':<10} {'Trap (Num)':<15} {'Simp (Num)':<15} {'Trap (Exato)':<15} {'Simp (Exato)':<15}")
        # print("-"*90)
        
        for h in h_valores:
            # Criar vetor x
            n = int(1.0 / h)
            x_vec = np.linspace(0, 1, n + 1)
            
            # Pegar solução numérica (RK4) e exata
            y_numerica = resultados[Nu][h]['rk4']['y_numerica']
            y_exata = resultados[Nu][h]['rk4']['y_exata']
            
            # Calcular fluxos da solução numérica
            fluxo_trap_num, fluxo_simp_num = calcularFluxoCalor(x_vec, y_numerica, Nu)
            
            # Calcular fluxos da solução exata
            fluxo_trap_ex, fluxo_simp_ex = calcularFluxoCalor(x_vec, y_exata, Nu)
            
            # Imprimir resultados
            print(f"{h:<10.4f} {fluxo_trap_num:<15.8f} {fluxo_simp_num:<15.8f} {fluxo_trap_ex:<15.8f} {fluxo_simp_ex:<15.8f}")
        
        print()

# =============================================================================
# TAREFA NUMÉRICA 2
# =============================================================================

def solucaoExataAfim(x, alpha, beta, Nu, f0):
    sqrt_nu = np.sqrt(Nu)

    # Solução particular
    y_p = f0 / Nu

    # Ajustar condições
    alpha_h = alpha - y_p
    beta_h = beta - y_p
    
    # Calcular exponenciais
    exp_neg = np.exp(-sqrt_nu)
    exp_pos = np.exp(sqrt_nu)
    
    # Resolver para B
    B = (beta_h - alpha_h * exp_neg) / (exp_pos - exp_neg)
    
    # Resolver para A
    A = alpha_h - B
    
    # Calcular y_ex(x)
    y_ex = A * np.exp(-sqrt_nu * x) + B * np.exp(sqrt_nu * x) + y_p
    
    return y_ex

def tarefaNumerica2(h, Nu, metodo, f0, alpha=0.1, beta=0.5, epsilon=0.0):
    # Parâmetros do teste
    n = int(1 / h)
    x_vec = np.linspace(0, 1, n + 1)

    # Solução exata
    y_exata = solucaoExataAfim(x_vec, alpha, beta, Nu, f0)
    
    # Solução numérica via método de disparo
    y_numerica = metodoDisparo(x_vec, alpha, beta, Nu, f0, metodo)

    # Cálculo do erro
    erro = calcularErro(y_numerica, y_exata)

    return y_numerica, y_exata, erro

def casosTarefaNumerica2():
    # Parâmetros para teste
    Nu = 1 
    f0_valores = [0, 1, 2, 4]
    h_valores = [0.5, 0.05, 0.005, 0.0005]
    metodos = ['euler_exp', 'euler_imp', 'trap_exp', 'trap_imp', 'rk4']
    
    resultados = {}
    
    for f0 in f0_valores:
        resultados[f0] = {}
        for h in h_valores:
            resultados[f0][h] = {}
            for metodo in metodos:
                y_num, y_ex, erro = tarefaNumerica2(h, Nu, metodo, f0)
                resultados[f0][h][metodo] = {
                    'erro': erro,
                    'y_numerica': y_num,
                    'y_exata': y_ex
                }
    
    return resultados

def plotarGraficosConvergencia2(resultados):
    Nu = 1
    f0_valores = [0, 1, 2, 4]
    h_valores = [0.5, 0.05, 0.005, 0.0005]
    metodos = ['euler_exp', 'euler_imp', 'trap_exp', 'trap_imp', 'rk4']
    
    nomes_metodos = {
        'euler_exp': 'Euler Explícito',
        'euler_imp': 'Euler Implícito',
        'trap_exp': 'Trapézio Explícito',
        'trap_imp': 'Trapézio Implícito',
        'rk4': 'Runge-Kutta 4'
    }
    
    # Criar uma figura com subplots (um para cada f0)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for idx, f0 in enumerate(f0_valores):
        ax = axs[idx]
        
        # Para cada método, coletar erros e plotar
        for metodo in metodos:
            erros = []
            for h in h_valores:
                erro = resultados[f0][h][metodo]['erro']
                erros.append(erro)
            
            # Plotar em escala log-log
            ax.loglog(h_valores, erros, 'o-', label=nomes_metodos[metodo])
        
        ax.set_xlabel('h (tamanho do passo)')
        ax.set_ylabel('Erro')
        ax.set_title(f'Nu = {Nu}, f0 = {f0}')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergencia_tarefa2.png', dpi=300, bbox_inches='tight')
    plt.show()

def plotarSolucoesTarefa2(resultados):
    f0_valores = [0, 1, 2, 4]
    h = 0.0005
    
    # Criar vetor x
    n = int(1.0 / h)
    x_vec = np.linspace(0, 1, n + 1)
    
    plt.figure(figsize=(10, 6))
    
    for f0 in f0_valores:
        # Pegar solução numérica (RK4)
        y = resultados[f0][h]['rk4']['y_numerica']
        plt.plot(x_vec, y, label=f'f₀ = {f0}', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Soluções para diferentes valores de f₀ (h = 0.0005, Nu = 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('solucoes_tarefa2.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# TAREFA NUMÉRICA 3
# =============================================================================

def metodoDisparoNaoLinear(x_vec, alpha, beta, Nu, f0, epsilon, t_inicial=0.0, max_iter=50, tol=1e-8):
    t = t_inicial
    
    for iteracao in range(max_iter):
        # Resolver problema não-linear com y'(0) = t
        y = rk4_nao_linear(x_vec, alpha, t, Nu, f0, epsilon)
        
        # Verificar convergência: y(1) deve ser igual a beta
        residuo = y[-1] - beta
        
        if abs(residuo) < tol:
            print(f"  ✅ Convergiu em {iteracao+1} iterações (t = {t:.6f})")
            return y, t
        
        # Resolver problema linearizado para calcular dy/dt
        z = rk4_linearizado(x_vec, 0.0, 1.0, y, Nu, epsilon)
        
        # Atualização de Newton: t_novo = t - residuo / z(1)
        if abs(z[-1]) < 1e-14:
            print("  ❌ Divisão por zero no método de Newton")
            break
        
        t = t - residuo / z[-1]
    
    print(f"  ⚠️ Não convergiu após {max_iter} iterações")
    return y, t