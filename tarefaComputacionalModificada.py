import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simpson_rule
from scipy.integrate import trapezoid

# =============================================================================
# FUNÇÕES DO PROBLEMA 
# =============================================================================

def p(x):
    return 1.0

def r(x, Nu):
    return Nu

def f(x, f0):
    return f0

def b(x):
    return 1.0

def g(x, y):
    return b(x) * (1 + y)**4

def dg_dy(x, y):
    return b(x) * 4 * (1 + y)**3

# =============================================================================
# MÉTODOS NUMÉRICOS (1) - TAREFA COMPUTACIONAL 1
# =============================================================================

def euler_explicito(x_vec, y0, dy0, Nu=1.0, f0=0.0):
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)

    y[0] = y0
    z[0] = dy0

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
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    
    y[0] = y0
    z[0] = dy0
    
    for k in range(n):
        x_k1 = x_vec[k + 1]
        p_k1 = 1.0  # p(x_k1)
        r_k1 = Nu    # r(x_k1, Nu)
        f_k1 = f0    # f(x_k1, f0)
        
        # Sistema: A * [y[k+1], z[k+1]]^T = b
        # Equação 1: y[k+1] - h*z[k+1] = y[k]
        # Equação 2: -(h*r/p)*y[k+1] + z[k+1] = z[k] + h*f/p
        
        coef_y = h * r_k1 / p_k1  

        A = np.array([[1.0, -h],
              [-coef_y, 1.0]])  

        b = np.array([y[k],
              z[k] - h * f_k1 / p_k1]) 
                
        # Verificar determinante antes de resolver
        det = np.linalg.det(A)
        if abs(det) < 1e-14:
            print(f"Matriz singular em k={k}, det={det:.2e}")

            y[k+1] = y[k] + h * z[k]
            dz_dx = (r_k1 * y[k] - f_k1) / p_k1
            z[k+1] = z[k] + h * dz_dx
        else:
            sol = np.linalg.solve(A, b)
            y[k + 1] = sol[0]
            z[k + 1] = sol[1]
    
    return y


def trapezio_explicito(x_vec, y0, dy0, Nu=1.0, f0=0.0):
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
    n = len(x_vec) - 1
    h = x_vec[1] - x_vec[0]
    
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    
    y[0] = y0
    z[0] = dy0
    
    for k in range(n):
        x_k = x_vec[k]
        x_k1 = x_vec[k + 1]
        
        p_k = 1.0
        p_k1 = 1.0
        r_k = Nu
        r_k1 = Nu
        f_k = f0
        f_k1 = f0
        
        # Calcular dz_k (explícito)
        dz_k = (r_k * y[k] - f_k) / p_k
        
        # Sistema: A * [y[k+1], z[k+1]]^T = b
        coef_y = h/2 * r_k1 / p_k1 

        A = np.array([[1.0, -h/2],
              [-coef_y, 1.0]])  

        b = np.array([y[k] + (h/2) * z[k],
              z[k] + (h/2) * dz_k - (h/2) * f_k1 / p_k1])
        
        # Verificar determinante antes de resolver
        det = np.linalg.det(A)
        if abs(det) < 1e-14:
            print(f"  ⚠️  Matriz singular em k={k}, det={det:.2e}")

            y_pred = y[k] + h * z[k]
            z_pred = z[k] + h * dz_k
            dz_k1 = (r_k1 * y_pred - f_k1) / p_k1
            y[k+1] = y[k] + (h/2) * (z[k] + z_pred)
            z[k+1] = z[k] + (h/2) * (dz_k + dz_k1)
        else:
            sol = np.linalg.solve(A, b)
            y[k + 1] = sol[0]
            z[k + 1] = sol[1]
    
    return y

def runge_kutta_4(x_vec, y0, dy0, Nu=1.0, f0=0.0):
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
# MÉTODOS NUMÉRICOS (2) - TAREFA COMPUTACIONAL 2
# =============================================================================

def rk4_nao_linear(x_vec, y0, dy0, Nu=1.0, f0=0.0, epsilon=1.0):
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
        x_temp = x_k + h/2
        z_temp = z_k + h/2 * k1_z
        w_temp = w_k + h/2 * k1_w
        # Interpolar y para o ponto médio
        y_temp = (y_vec[k] + y_vec[k+1]) / 2 
        
        k2_z = w_temp
        k2_w = (r(x_temp, Nu)*z_temp + epsilon*dg_dy(x_temp, y_temp)*z_temp) / p(x_temp)
        
        # k3
        z_temp = z_k + h/2 * k2_z
        w_temp = w_k + h/2 * k2_w
        # y_temp já foi calculado
        
        k3_z = w_temp
        k3_w = (r(x_temp, Nu)*z_temp + epsilon*dg_dy(x_temp, y_temp)*z_temp) / p(x_temp)
        
        # k4
        x_temp = x_k + h
        z_temp = z_k + h * k3_z
        w_temp = w_k + h * k3_w
        y_temp = y_vec[k+1] 
        
        k4_z = w_temp
        k4_w = (r(x_temp, Nu)*z_temp + epsilon*dg_dy(x_temp, y_temp)*z_temp) / p(x_temp)
        
        # Atualizar
        z[k+1] = z_k + h/6 * (k1_z + 2*k2_z + 2*k3_z + k4_z)
        w[k+1] = w_k + h/6 * (k1_w + 2*k2_w + 2*k3_w + k4_w)
    
    return z

# =============================================================================
# RESULTADO TAREFAS COMPUTACIONAIS 1 e 2
# =============================================================================

def teste_tarefas_computacionais():
    print("="*70)
    print("TESTANDO OS MÉTODOS NUMÉRICOS (TAREFAS COMPUTACIONAIS 1 E 2)")
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
    print(f"  n = {n} pontos (h = {1.0/n:.3f})")
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
    
    # Teste 1: epsilon = 0 
    epsilon = 0.0
    y_nl_0 = rk4_nao_linear(x_vec, y0, dy0, Nu, f0, epsilon)
    print(f"RK4 Não-Linear (ε=0): y(0.5) = {y_nl_0[n//2]:.6f}  |  y(1) = {y_nl_0[-1]:.6f}")
    print(f"  → Diferença com RK4 linear: {abs(y_nl_0[-1] - y_rk4[-1]):.2e}")
    
    # Teste 2: epsilon = 1
    epsilon = 1.0
    y_nl_1 = rk4_nao_linear(x_vec, y0, dy0, Nu, f0, epsilon)
    print(f"RK4 Não-Linear (ε=1): y(0.5) = {y_nl_1[n//2]:.6f}  |  y(1) = {y_nl_1[-1]:.6f}")
    
    # Testar método LINEARIZADO
    print("\n" + "-"*70)
    print("MÉTODO LINEARIZADO (Tarefa Computacional 3 - parte do Newton)")
    print("-"*70)
    
    # Condições para o linearizado
    z0 = 0.0
    dz0 = 1.0
    
    z_lin = rk4_linearizado(x_vec, z0, dz0, y_nl_1, Nu, epsilon)
    print(f"RK4 Linearizado:     z(0.5) = {z_lin[n//2]:.6f}  |  z(1) = {z_lin[-1]:.6f}")
    
    # Verificação: z deve começar em 0
    print(f"  → z(0) = {z_lin[0]:.6f} (deve ser 0.0)")
    
    print("\n" + "="*70)
    print("TESTES DAS TAREFAS COMPUTACIONAIS CONCLUÍDOS!")
    print("="*70)

# =============================================================================
# TAREFA NUMÉRICA 1
# =============================================================================

def metodoDisparo(x_vec, alpha, beta, Nu, f0, metodo):
    # Definição do método
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
    if abs(w[-1]) < 1e-10:
        print(f"Aviso: w(1) = {w[-1]:.2e} muito pequeno\n")
        print(f"Usando método alternativo para h = {x_vec[1]-x_vec[0]}")
        
        # Mas vamos tentar com um epsilon pequeno
        w_final = w[-1]
        if abs(w_final) < 1e-14:
            # Caso extremo: usar solução linear
            y = alpha + (beta - alpha) * x_vec
            return y
        
    # Determinar coeficientes a e b
    # y(0) = a*v(0) + b*w(0) + u_p(0)
    # alpha = a*1 + b*0 + u_p(0) -> a = alpha - u_p(0)
    a = alpha - u_p[0]
    
    # y(1) = a*v(1) + b*w(1) + u_p(1)
    # beta = a*v(1) + b*w(1) + u_p(1)
    # b = (beta - a*v(1) - u_p(1)) / w(1)
    b = (beta - a * v[-1] - u_p[-1]) / w[-1]
    
    # Solução final
    y = a * v + b * w + u_p
    
    return y


def solucaoExata(x, alpha, beta, Nu):
    sqrt_nu = np.sqrt(Nu)
    
    exp_neg = np.exp(-sqrt_nu)
    exp_pos = np.exp(sqrt_nu)
    
    den = exp_pos - exp_neg
    if abs(den) < 1e-14:
        return alpha + (beta - alpha) * x

    B = (beta - alpha * exp_neg) / den

    A = alpha - B

    y_ex = A * np.exp(-sqrt_nu * x) + B * np.exp(sqrt_nu * x)
    
    return y_ex

def calcularErro(y_numerica, y_exata):
    erroQuadratico = np.sqrt(np.mean((y_numerica - y_exata)**2))
    return erroQuadratico

def tarefaNumerica1(h, Nu, metodo, alpha=0.1, beta=0.5, f0=0.0):

    n = int(1.0 / h)
    x_vec = np.linspace(0, 1, n + 1)

    y_exata = solucaoExata(x_vec, alpha, beta, Nu)
  
    y_numerica = metodoDisparo(x_vec, alpha, beta, Nu, f0, metodo)

    erro = calcularErro(y_numerica, y_exata)

    return y_numerica, y_exata, erro

def casosTarefaNumerica1():
    Nu_valores = [1, 16, 256]
    h_valores = [0.5, 0.05, 0.005, 0.0005]
    metodos = ['euler_exp', 'euler_imp', 'trap_exp', 'trap_imp', 'rk4']

    resultados = {}
    
    print("\nExecutando Tarefa Numérica 1 (Cálculo de Erros)...")
    
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
    print("Cálculo de erros da Tarefa 1 concluído.")
    return resultados

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
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Tarefa 1: Análise de Convergência (Erro RMS vs h)', fontsize=16)
    
    for idx, Nu in enumerate(Nu_valores):
        ax = axs[idx]

        for metodo in metodos:
            erros = []
            for h in h_valores:
                erro = resultados[Nu][h][metodo]['erro']
                erros.append(erro)

            ax.loglog(h_valores, erros, 'o-', label=nomes_metodos[metodo])

        if 'rk4' in metodos:
            erros_rk4 = [resultados[Nu][h]['rk4']['erro'] for h in h_valores]
            h_ref = np.array(h_valores)
            ax.loglog(h_ref, (erros_rk4[1]/h_ref[1]**4) * h_ref**4, 'k:', label='O(h⁴)')
            ax.loglog(h_ref, (erros_rk4[1]/h_ref[1]**2) * h_ref**2, 'k--', label='O(h²)')
            ax.loglog(h_ref, (erros_rk4[1]/h_ref[1]**1) * h_ref**1, 'k-.', label='O(h¹)')


        ax.set_xlabel('h (tamanho do passo)')
        if idx == 0:
            ax.set_ylabel('Erro RMS')
        ax.set_title(f'Nu = {Nu}')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.set_xticks(h_valores)
        ax.set_xticklabels(map(str, h_valores))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('convergencia_tarefa1.png', dpi=300, bbox_inches='tight')
    print("Gráfico 'convergencia_tarefa1.png' salvo.")
    plt.show()

def calcularFluxoCalor(x_vec, y, Nu):
    integrando = Nu * y  
    
    fluxo_trapezios = trapezoid(integrando, x_vec)

    fluxo_simpson = simpson_rule(integrando, x=x_vec)
    
    return fluxo_trapezios, fluxo_simpson

def analisarFluxosCalor(resultados):
    Nu_valores = [1, 16, 256]
    h_valores = [0.5, 0.05, 0.005, 0.0005]
    
    print("\n" + "="*90)
    print("TAREFA 1: ANÁLISE DE FLUXO DE CALOR (q_conv)")
    print("Comparando integrais da solução Exata vs Numérica (RK4)")
    print("="*90)
    
    for Nu in Nu_valores:
        print(f"\nNu = {Nu}")
        print("─"*90)
        print(f"{'h':<10} {'Trap (Num)':<15} {'Simp (Num)':<15} {'Trap (Exato)':<15} {'Simp (Exato)':<15}")
        print("─"*90)
        
        for h in h_valores:
 
            n = int(1.0 / h)
            if n % 2 != 0: n += 1 
            x_vec = np.linspace(0, 1, n + 1)

            y_numerica, y_exata, _ = tarefaNumerica1(x_vec[1]-x_vec[0], Nu, 'rk4')

            fluxo_trap_num, fluxo_simp_num = calcularFluxoCalor(x_vec, y_numerica, Nu)

            fluxo_trap_ex, fluxo_simp_ex = calcularFluxoCalor(x_vec, y_exata, Nu)

            print(f"{h:<10.4f} {fluxo_trap_num:<15.8f} {fluxo_simp_num:<15.8f} {fluxo_trap_ex:<15.8f} {fluxo_simp_ex:<15.8f}")
        
        print("─"*90)

# =============================================================================
# TAREFA NUMÉRICA 2
# =============================================================================

def solucaoExataAfim(x, alpha, beta, Nu, f0):
    sqrt_nu = np.sqrt(Nu)

    y_p = f0 / Nu

    alpha_h = alpha - y_p
    beta_h = beta - y_p
    
    exp_neg = np.exp(-sqrt_nu)
    exp_pos = np.exp(sqrt_nu)

    den = exp_pos - exp_neg
    if abs(den) < 1e-14:
        return (alpha_h + (beta_h - alpha_h) * x) + y_p

    B = (beta_h - alpha_h * exp_neg) / den
    
    A = alpha_h - B

    y_ex = A * np.exp(-sqrt_nu * x) + B * np.exp(sqrt_nu * x) + y_p
    
    return y_ex

def tarefaNumerica2(h, Nu, metodo, f0, alpha=0.1, beta=0.5):

    n = int(1.0 / h)
    x_vec = np.linspace(0, 1, n + 1)

    y_exata = solucaoExataAfim(x_vec, alpha, beta, Nu, f0)
    
    y_numerica = metodoDisparo(x_vec, alpha, beta, Nu, f0, metodo)

    erro = calcularErro(y_numerica, y_exata)

    return y_numerica, y_exata, erro

def casosTarefaNumerica2():
    Nu = 1 
    f0_valores = [0, 1, 2, 4]
    h_valores = [0.5, 0.05, 0.005, 0.0005]
    metodos = ['euler_exp', 'euler_imp', 'trap_exp', 'trap_imp', 'rk4']
    
    resultados = {}
    
    print("\nExecutando Tarefa Numérica 2 (Cálculo de Erros com f0 != 0)...")
    
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
                
    print("Cálculo de erros da Tarefa 2 concluído.")
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

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    fig.suptitle('Tarefa 2: Análise de Convergência (Erro RMS vs h) para Nu = 1', fontsize=16)
    axs = axs.flatten()
    
    for idx, f0 in enumerate(f0_valores):
        ax = axs[idx]
        
        for metodo in metodos:
            erros = []
            for h in h_valores:
                erro = resultados[f0][h][metodo]['erro']
                erros.append(erro)
        
            ax.loglog(h_valores, erros, 'o-', label=nomes_metodos[metodo])
        
        ax.set_xlabel('h (tamanho do passo)')
        ax.set_ylabel('Erro RMS')
        ax.set_title(f'f0 = {f0}')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.set_xticks(h_valores)
        ax.set_xticklabels(map(str, h_valores))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('convergencia_tarefa2.png', dpi=300, bbox_inches='tight')
    print("Gráfico 'convergencia_tarefa2.png' salvo.")
    plt.show()

def plotarSolucoesTarefa2(resultados):
    f0_valores = [0, 1, 2, 4]
    h = 0.0005
    
    n = int(1.0 / h)
    x_vec = np.linspace(0, 1, n + 1)
    
    plt.figure(figsize=(10, 6))
    
    for f0 in f0_valores:
        y = resultados[f0][h]['rk4']['y_numerica']
        plt.plot(x_vec, y, label=f'f₀ = {f0}', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Tarefa 2: Soluções para diferentes valores de f₀ (h = 0.0005, Nu = 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('solucoes_tarefa2.png', dpi=300, bbox_inches='tight')
    print("Gráfico 'solucoes_tarefa2.png' salvo.")
    plt.show()

# =============================================================================
# TAREFA NUMÉRICA 3
# =============================================================================

def metodoDisparoNaoLinear(x_vec, alpha, beta, Nu, f0, epsilon, t_inicial=0.0, max_iter=50, tol=1e-8):
    t = t_inicial
    y = np.zeros_like(x_vec)
    
    for iteracao in range(max_iter):
        y = rk4_nao_linear(x_vec, alpha, t, Nu, f0, epsilon)
   
        residuo = y[-1] - beta
        
        if abs(residuo) < tol:
            print(f"  ✅ Convergiu em {iteracao+1} iterações (t = {t:.6f})")
            return y, t
        
        z = rk4_linearizado(x_vec, 0.0, 1.0, y, Nu, epsilon)
        
        if abs(z[-1]) < 1e-14:
            print(f"  ❌ Divisão por zero no método de Newton (z(1) = {z[-1]:.2e})")
            break
        
        t = t - residuo / z[-1]
    
    print(f"  ⚠️ Não convergiu após {max_iter} iterações (Resíduo = {residuo:.2e})")
    return y, t

def calcularCalorRadiacao(x_vec, y, epsilon):

    # Integrando: epsilon * b(x) * (1 + y(x))^4
    integrando = epsilon * g(x_vec, y) # g(x,y) = b(x) * (1+y)^4

    q_rad_trapezios = trapezoid(integrando, x_vec)

    q_rad_simpson = simpson_rule(integrando, x=x_vec)
    
    return q_rad_trapezios, q_rad_simpson

def tarefaNumerica3():

    Nu_valores = [1, 16, 256]
    f0 = 1
    h = 0.0005
    alpha = 0.1
    beta = 0.5

    n = int(1.0 / h)
    if n % 2 != 0: n += 1 
    x_vec = np.linspace(0, 1, n + 1)
    
    print("\n" + "="*90)
    print("TAREFA NUMÉRICA 3: ANÁLISE NÃO-LINEAR (ε=1) vs LINEAR (ε=0)")
    print("="*90)
    print(f"\nParâmetros: f₀ = {f0}, h = {h}, α = {alpha}, β = {beta}")
    
    resultados = {}
    
    for Nu in Nu_valores:
        print(f"\n{'─'*90}")
        print(f"Nu = {Nu}")
        print(f"{'─'*90}")
        
        resultados[Nu] = {}

        print(f"\n[LINEAR] Resolvendo com método de disparo (ε = 0)...")
        y_linear = metodoDisparo(x_vec, alpha, beta, Nu, f0, 'rk4')

        q_conv_trap_lin, q_conv_simp_lin = calcularFluxoCalor(x_vec, y_linear, Nu)
        
        resultados[Nu]['linear'] = {
            'y': y_linear,
            'q_conv_trap': q_conv_trap_lin,
            'q_conv_simp': q_conv_simp_lin,
            'q_rad_trap': 0.0,
            'q_rad_simp': 0.0
        }

        print(f"\n[NÃO-LINEAR] Resolvendo com método de disparo não-linear (ε = 1)...")

        t_chute = 0.0
        print(f"  (Usando chute inicial t = {t_chute:.6f})")

        y_nao_linear, t_final = metodoDisparoNaoLinear(
            x_vec, alpha, beta, Nu, f0, 
            epsilon=1.0, 
            t_inicial=t_chute,
            max_iter=100,  
            tol=1e-6       
        )

        q_conv_trap_nl, q_conv_simp_nl = calcularFluxoCalor(x_vec, y_nao_linear, Nu)

        q_rad_trap_nl, q_rad_simp_nl = calcularCalorRadiacao(x_vec, y_nao_linear, epsilon=1.0)
        
        resultados[Nu]['nao_linear'] = {
            'y': y_nao_linear,
            'q_conv_trap': q_conv_trap_nl,
            'q_conv_simp': q_conv_simp_nl,
            'q_rad_trap': q_rad_trap_nl,
            'q_rad_simp': q_rad_simp_nl,
            't_final': t_final
        }
        
        print(f"\n{'─'*90}")
        print(f"RESULTADOS DE FLUXO DE CALOR PARA Nu = {Nu}")
        print(f"{'─'*90}")
        print(f"{'Método':<20} {'q_conv (Trap)':<18} {'q_conv (Simp)':<18} {'q_rad (Trap)':<18} {'q_rad (Simp)':<18}")
        print("─"*90)
        print(f"{'Linear (ε=0)':<20} {q_conv_trap_lin:<18.8f} {q_conv_simp_lin:<18.8f} {0.0:<18.8f} {0.0:<18.8f}")
        print(f"{'Não-linear (ε=1)':<20} {q_conv_trap_nl:<18.8f} {q_conv_simp_nl:<18.8f} {q_rad_trap_nl:<18.8f} {q_rad_simp_nl:<18.8f}")
        print("─"*90)
        q_total_trap = q_conv_trap_nl + q_rad_trap_nl
        q_total_simp = q_conv_simp_nl + q_rad_simp_nl
        print(f"{'TOTAL (ε=1)':<20} {'-':<18} {'-':<18} {q_total_trap:<18.8f} {q_total_simp:<18.8f}")
        print(f"{'─'*90}")
    
    return resultados

def plotarComparacaoLinearNaoLinear(resultados):

    Nu_valores = [1, 16, 256]
    h = 0.0005

    n = int(1.0 / h)
    if n % 2 != 0: n += 1 
    x_vec = np.linspace(0, 1, n + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Tarefa 3: Comparação das Soluções y(x) [Linear vs Não-Linear]', fontsize=16)
    
    for idx, Nu in enumerate(Nu_valores):
        ax = axs[idx]

        y_linear = resultados[Nu]['linear']['y']
        ax.plot(x_vec, y_linear, 'b-', linewidth=2, label='Linear (ε = 0)')

        y_nao_linear = resultados[Nu]['nao_linear']['y']
        ax.plot(x_vec, y_nao_linear, 'r--', linewidth=2, label='Não-linear (ε = 1)')
        
        ax.set_xlabel('x')
        if idx == 0:
            ax.set_ylabel('y(x)')
        ax.set_title(f'Nu = {Nu}, f₀ = 1')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparacao_linear_nao_linear.png', dpi=300, bbox_inches='tight')
    print("Gráfico 'comparacao_linear_nao_linear.png' salvo.")
    plt.show()

#def plotarDiferencaLinearNaoLinear(resultados):

    # Nu_valores = [1, 16, 256]
    # h = 0.0005
    
    # n = int(1.0 / h)
    # if n % 2 != 0: n += 1
    # x_vec = np.linspace(0, 1, n + 1)
    
    # plt.figure(figsize=(10, 6))
    
    # for Nu in Nu_valores:

    #     y_linear = resultados[Nu]['linear']['y']
    #     y_nao_linear = resultados[Nu]['nao_linear']['y']
    #     diferenca = y_nao_linear - y_linear

    #     print(f"\nNu = {Nu}")
    #     print(f"  Diferença máxima: {np.max(np.abs(diferenca)):.6e}")
    #     print(f"  Diferença em x=0.5: {diferenca[len(diferenca)//2]:.6e}")
        
    #     plt.semilogy(x_vec, np.abs(diferenca) + 1e-16, linewidth=2, label=f'Nu = {Nu}')
    
    # plt.xlabel('x')
    # plt.ylabel('|y_não_linear(x) - y_linear(x)|')
    # plt.title('Tarefa 3: Diferença Absoluta entre soluções (f₀ = 1, escala log)')
    # plt.legend()
    # plt.grid(True, alpha=0.3, which='both')
    # plt.tight_layout()
    # plt.savefig('diferenca_linear_nao_linear.png', dpi=300, bbox_inches='tight')
    # print("\nGráfico 'diferenca_linear_nao_linear.png' salvo.")
    # plt.show()
    
    # # GRÁFICO ADICIONAL: Diferença relativa
    # plt.figure(figsize=(10, 6))
    
    # for Nu in Nu_valores:
    #     y_linear = resultados[Nu]['linear']['y']
    #     y_nao_linear = resultados[Nu]['nao_linear']['y']
        
    #     diferenca_relativa = np.abs((y_nao_linear - y_linear) / (np.abs(y_linear) + 1e-16))
        
    #     plt.semilogy(x_vec, diferenca_relativa, linewidth=2, label=f'Nu = {Nu}')
    
    # plt.xlabel('x')
    # plt.ylabel('Erro relativo: |y_nl - y_l| / |y_l|')
    # plt.title('Tarefa 3: Diferença Relativa entre soluções (f₀ = 1)')
    # plt.legend()
    # plt.grid(True, alpha=0.3, which='both')
    # plt.tight_layout()
    # plt.savefig('diferenca_relativa_linear_nao_linear.png', dpi=300, bbox_inches='tight')
    # print("Gráfico 'diferenca_relativa_linear_nao_linear.png' salvo.")
    # plt.show()
    
# =============================================================================
# RESULTADOS EM VALORES E GRÁFICOS
# =============================================================================

if __name__ == "__main__":
    
    # 1. TESTE DAS TAREFAS COMPUTACIONAIS (MÉTODOS BÁSICOS)
    teste_tarefas_computacionais()

    # 2. RESULTADOS DA TAREFA NUMÉRICA 1 (Caso Linear, f0=0)
    print("\n\n" + "="*90)
    print("INICIANDO TAREFA NUMÉRICA 1")
    print("="*90)
    resultados_tn1 = casosTarefaNumerica1()
    plotarGraficosConvergencia(resultados_tn1)
    analisarFluxosCalor(resultados_tn1)

    # 3. RESULTADOS DA TAREFA NUMÉRICA 2 (Caso Linear, f0 != 0)
    print("\n\n" + "="*90)
    print("INICIANDO TAREFA NUMÉRICA 2")
    print("="*90)
    resultados_tn2 = casosTarefaNumerica2()
    plotarGraficosConvergencia2(resultados_tn2)
    plotarSolucoesTarefa2(resultados_tn2)

    # 4. RESULTADOS DA TAREFA NUMÉRICA 3 (Caso Não-Linear, f0 != 0)
    print("\n\n" + "="*90)
    print("INICIANDO TAREFA NUMÉRICA 3")
    print("="*90)
    resultados_tn3 = tarefaNumerica3()
    plotarComparacaoLinearNaoLinear(resultados_tn3)
    #plotarDiferencaLinearNaoLinear(resultados_tn3)

    print("\n\n" + "="*90)
    print("TODAS AS TAREFAS FORAM EXECUTADAS E OS RESULTADOS FORAM GERADOS.")
    print("="*90)